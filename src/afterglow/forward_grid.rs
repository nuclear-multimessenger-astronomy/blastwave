use crate::constants::*;
use crate::hydro::tools::Tool;
use crate::afterglow::blast::Blast;
use crate::afterglow::eats::EATS;
use crate::afterglow::models::{Dict, RadiationModel, CachedSyncParams};

/// Pre-computed forward-mapping grid for fast luminosity computation.
///
/// Works for both on-axis and off-axis viewing angles.
/// On-axis (theta_v < 1e-6): single phi=0, domega = dcos_theta * 2π.
/// Off-axis: nphi cells in [0, π], domega = dcos_theta * dphi * 2 (reflection symmetry).
///
/// Stores log₂ values for fast interpolation (1 exp2 vs 6 ln/exp per cell).
pub struct ForwardGrid {
    /// log₂ of observer-frame times (de-redshifted): [ncell][nt]
    lg2_t_obs: Vec<Vec<f64>>,
    /// log₂ of dL/dOmega: [ncell][nt] (NEG_INFINITY for zero)
    lg2_dl_domega: Vec<Vec<f64>>,
    /// Solid angle weights per cell (includes phi integration)
    domega: Vec<f64>,
    /// Per-cell min lg2_t for fast range check
    lg2_t_min: Vec<f64>,
    /// Per-cell max lg2_t for fast range check
    lg2_t_max: Vec<f64>,
}

/// Maximum number of time substeps to keep from the hydro grid.
const MAX_NT: usize = 300;
/// Number of phi cells for off-axis adaptive integration (covers [0, π]).
const NPHI_OFFAXIS: usize = 8;

/// Subsample the time grid to at most MAX_NT indices (uniform fallback).
fn subsample_time_indices(nt: usize) -> Vec<usize> {
    if nt <= MAX_NT {
        (0..nt).collect()
    } else {
        let stride = nt / MAX_NT;
        let mut indices: Vec<usize> = (0..nt).step_by(stride).collect();
        if *indices.last().unwrap() != nt - 1 {
            indices.push(nt - 1);
        }
        indices
    }
}

/// Per-cell adaptive time indices that concentrate points near deceleration time.
///
/// Finds t_dec (where beta_gamma_sq drops to max/e), then weights 2× within
/// 1 decade of t_dec. CDF-based sampling distributes MAX_NT indices.
fn adaptive_time_indices(nt: usize, y_data: &[Vec<Vec<f64>>], j: usize, t_data: &[f64]) -> Vec<usize> {
    if nt <= MAX_NT {
        return (0..nt).collect();
    }

    let bg_sq = &y_data[2][j];

    // Find deceleration time: where beta_gamma_sq drops to max/e
    let max_bg = bg_sq.iter().copied().fold(0.0f64, f64::max);
    if max_bg <= 0.0 {
        return subsample_time_indices(nt);
    }

    let threshold = max_bg / std::f64::consts::E;
    let mut t_dec_idx = 0;
    for k in 0..nt {
        if bg_sq[k] >= threshold {
            t_dec_idx = k;
        }
    }
    let t_dec = t_data[t_dec_idx];

    // Build weight function: 2× within 1 decade of t_dec, 1× elsewhere
    let lg_t_dec = if t_dec > 0.0 { t_dec.log10() } else { 0.0 };
    let mut weights = vec![0.0f64; nt];
    for k in 0..nt {
        let lg_t = if t_data[k] > 0.0 { t_data[k].log10() } else { 0.0 };
        let dist = (lg_t - lg_t_dec).abs();
        weights[k] = if dist < 1.0 { 2.0 } else { 1.0 };
    }

    // Build CDF
    let mut cdf = vec![0.0f64; nt + 1];
    for k in 0..nt {
        cdf[k + 1] = cdf[k] + weights[k];
    }
    let total = cdf[nt];
    if total <= 0.0 {
        return subsample_time_indices(nt);
    }

    // Invert CDF to get MAX_NT indices
    let mut indices = Vec::with_capacity(MAX_NT);
    indices.push(0); // Always include first

    for i in 1..MAX_NT - 1 {
        let target = total * i as f64 / (MAX_NT - 1) as f64;
        // Find the interval in CDF
        let mut idx = 0;
        while idx < nt && cdf[idx + 1] < target {
            idx += 1;
        }
        if idx >= nt {
            idx = nt - 1;
        }
        // Avoid duplicates
        if indices.last().map_or(true, |&last| last != idx) {
            indices.push(idx);
        }
    }

    // Always include last
    if indices.last().map_or(true, |&last| last != nt - 1) {
        indices.push(nt - 1);
    }

    indices
}

/// Compute cell boundaries and dcos_theta weights for the theta grid.
fn compute_dcos_theta(theta_data: &[f64]) -> Vec<f64> {
    let ntheta = theta_data.len();
    let mut boundaries = vec![0.0; ntheta + 1];
    boundaries[0] = 0.0;
    for j in 1..ntheta {
        boundaries[j] = (theta_data[j - 1] + theta_data[j]) / 2.0;
    }
    boundaries[ntheta] = PI;
    (0..ntheta)
        .map(|j| boundaries[j].cos() - boundaries[j + 1].cos())
        .collect()
}

/// Compute the beta_gamma_sq threshold for skipping negligible cells.
fn compute_bg_threshold(y_data: &[Vec<Vec<f64>>], ntheta: usize) -> f64 {
    let max_bg_sq = (0..ntheta)
        .flat_map(|j| y_data[2][j].iter().copied())
        .fold(0.0f64, f64::max);
    max_bg_sq * 1e-6
}

/// Detect theta cells with identical hydro profiles.
/// Returns a map: representative[j] = index of representative cell for cell j.
/// Cells with the same representative share y_data (save array access cost).
fn detect_theta_groups(y_data: &[Vec<Vec<f64>>], ntheta: usize, bg_threshold: f64) -> Vec<usize> {
    let mut representative = Vec::with_capacity(ntheta);
    let rel_tol = 0.01; // 1% relative error threshold
    let nvar = y_data.len().min(6); // Compare first 6 variables

    'outer: for j in 0..ntheta {
        // Check against all earlier cells
        for &rep in &representative[..j] {
            if rep != j {
                // Only compare with actual representatives
                continue;
            }
        }

        // Check if this cell matches any earlier representative
        for prev_j in 0..j {
            let prev_rep = representative[prev_j];
            if prev_rep != prev_j {
                continue; // Only compare with actual representatives
            }

            // Compare all variables
            let nt = y_data[0][j].len();
            let mut matches = true;
            for v in 0..nvar {
                for k in 0..nt {
                    let a = y_data[v][j][k];
                    let b = y_data[v][prev_j][k];
                    let max_ab = a.abs().max(b.abs());
                    if max_ab > 0.0 && (a - b).abs() / max_ab > rel_tol {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    break;
                }
            }

            if matches {
                representative.push(prev_j);
                continue 'outer;
            }
        }

        // No match found — this cell is its own representative
        representative.push(j);
    }

    representative
}

/// Build the phi grid adaptive to Doppler beaming.
/// On-axis: single phi=0, dphi=2π (azimuthal symmetry).
/// Off-axis: NPHI_OFFAXIS cells concentrated where D³(phi) peaks.
///
/// Uses D³(phi) weight from the most-beamed theta cell to build a CDF,
/// then inverts it to place cells at equal-weight intervals.
fn build_phi_grid(theta_v: f64, theta_data: &[f64], y_data: &[Vec<Vec<f64>>]) -> (Vec<f64>, Vec<f64>) {
    if theta_v.abs() < 1e-6 {
        // On-axis: azimuthal symmetry, single cell at phi=0
        return (vec![0.0], vec![2.0 * PI]);
    }

    let nphi = NPHI_OFFAXIS;
    let ntheta = theta_data.len();

    // Find theta cell with max beta*gamma (most beamed)
    let mut best_j = 0;
    let mut best_bg = 0.0_f64;
    for j in 0..ntheta {
        let bg = y_data[2][j].iter().copied().fold(0.0f64, f64::max);
        if bg > best_bg {
            best_bg = bg;
            best_j = j;
        }
    }

    // Compute Gamma for the most-beamed cell (use peak bg_sq)
    let bg_sq = best_bg;
    let gamma = (1.0 + bg_sq).sqrt();
    let beta = (bg_sq / (1.0 + bg_sq)).sqrt();
    let theta_best = theta_data[best_j];

    // Sample D³(phi) on a fine grid of 100 points in [0, π]
    let n_fine = 100;
    let cos_tv = theta_v.cos();
    let sin_tv = theta_v.sin();
    let cos_th = theta_best.cos();
    let sin_th = theta_best.sin();

    let mut weights = vec![0.0f64; n_fine];
    let mut phi_fine = vec![0.0f64; n_fine];

    for i in 0..n_fine {
        let phi = (i as f64 + 0.5) * PI / n_fine as f64;
        phi_fine[i] = phi;

        // mu = cos(angle between radial direction and observer)
        let mu = cos_th * cos_tv + sin_th * phi.cos() * sin_tv;
        // Doppler factor D = 1 / (Gamma * (1 - beta * mu))
        let d_inv = gamma * (1.0 - beta * mu);
        if d_inv > 0.0 {
            let d = 1.0 / d_inv;
            weights[i] = d * d * d; // D³
        }
    }

    // Build CDF
    let dphi_fine = PI / n_fine as f64;
    let mut cdf = vec![0.0f64; n_fine + 1];
    for i in 0..n_fine {
        cdf[i + 1] = cdf[i] + weights[i] * dphi_fine;
    }
    let total = cdf[n_fine];

    if total <= 0.0 {
        // Fallback to uniform grid
        let dphi_base = PI / nphi as f64;
        let phis: Vec<f64> = (0..nphi)
            .map(|k| (k as f64 + 0.5) * dphi_base)
            .collect();
        let dphis: Vec<f64> = vec![dphi_base * 2.0; nphi];
        return (phis, dphis);
    }

    // Normalize CDF
    for v in cdf.iter_mut() {
        *v /= total;
    }

    // Invert CDF to get cell boundaries at equal-weight intervals
    let mut boundaries = vec![0.0f64; nphi + 1];
    boundaries[0] = 0.0;
    boundaries[nphi] = PI;

    for k in 1..nphi {
        let target = k as f64 / nphi as f64;
        // Find interval in CDF
        let mut idx = 0;
        while idx < n_fine && cdf[idx + 1] < target {
            idx += 1;
        }
        // Linear interpolation within the interval
        let frac = if cdf[idx + 1] > cdf[idx] {
            (target - cdf[idx]) / (cdf[idx + 1] - cdf[idx])
        } else {
            0.5
        };
        boundaries[k] = (idx as f64 + frac) * dphi_fine;
    }

    // Compute cell centers and dphi values
    let phis: Vec<f64> = (0..nphi)
        .map(|k| (boundaries[k] + boundaries[k + 1]) / 2.0)
        .collect();
    // Factor of 2 for [π, 2π] reflection symmetry
    let dphis: Vec<f64> = (0..nphi)
        .map(|k| (boundaries[k + 1] - boundaries[k]) * 2.0)
        .collect();

    (phis, dphis)
}

/// Convert linear values to log₂, mapping non-positive to NEG_INFINITY.
#[inline]
fn to_lg2(vals: &[f64]) -> Vec<f64> {
    vals.iter()
        .map(|&v| if v > 0.0 { v.log2() } else { f64::NEG_INFINITY })
        .collect()
}

/// Cached blast states for multi-frequency forward-grid construction.
///
/// Pre-computes frequency-independent blast states once, then builds
/// ForwardGrids per frequency with only radiation model evaluations.
pub struct BlastGridCache {
    /// Blast states per cell: [ncell][nt_sub]
    blasts: Vec<Vec<Blast>>,
    /// Observer-frame times per cell: [ncell][nt_sub]
    t_obs: Vec<Vec<f64>>,
    /// Solid angle weights per cell
    domega: Vec<f64>,
    /// Optional cached synchrotron params per cell: [ncell][nt_sub]
    cached_sync: Option<Vec<Vec<CachedSyncParams>>>,
}

impl BlastGridCache {
    /// Pre-compute all frequency-independent blast states.
    pub fn precompute(
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        eats: &EATS,
        tool: &Tool,
    ) -> Self {
        let ntheta = theta_data.len();
        let nt = t_data.len();
        let nvar = y_data.len();

        let all_dcos = compute_dcos_theta(theta_data);
        let bg_threshold = compute_bg_threshold(y_data, ntheta);
        let (phis, dphis) = build_phi_grid(theta_v, theta_data, y_data);
        let theta_groups = detect_theta_groups(y_data, ntheta, bg_threshold);

        let mut blasts_out = Vec::new();
        let mut t_obs_out = Vec::new();
        let mut domega_out = Vec::new();

        for j in 0..ntheta {
            let cell_max_bg = y_data[2][j].iter().copied().fold(0.0f64, f64::max);
            if cell_max_bg < bg_threshold {
                continue;
            }

            // Per-cell adaptive time indices
            let time_indices = adaptive_time_indices(nt, y_data, j, t_data);
            let nt_sub = time_indices.len();

            let theta = theta_data[j];
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Use representative cell's y_data for val extraction
            let rep = theta_groups[j];

            for (phi_idx, &phi) in phis.iter().enumerate() {
                let mu = cos_theta * theta_v.cos() + sin_theta * phi.cos() * theta_v.sin();

                let mut cell_blasts = Vec::with_capacity(nt_sub);
                let mut cell_tobs = Vec::with_capacity(nt_sub);

                for &k in &time_indices {
                    let r = y_data[4][j][k];
                    cell_tobs.push(t_data[k] - r * mu / C_SPEED);

                    let val = [
                        y_data[0][rep][k],
                        y_data[1][rep][k],
                        y_data[2][rep][k],
                        y_data[3][rep][k],
                        y_data[4][rep][k],
                        y_data[5][rep][k],
                        if nvar > 6 { y_data[6][rep][k] } else { 0.0 },
                        t_data[k],
                    ];

                    let mut blast = Blast::default();
                    eats.derive_blast(theta, phi, theta_v, &val, tool, &mut blast);
                    cell_blasts.push(blast);
                }

                blasts_out.push(cell_blasts);
                t_obs_out.push(cell_tobs);
                domega_out.push(all_dcos[j] * dphis[phi_idx]);
            }
        }

        BlastGridCache {
            blasts: blasts_out,
            t_obs: t_obs_out,
            domega: domega_out,
            cached_sync: None,
        }
    }

    /// Pre-compute all frequency-independent blast states with sync param caching.
    pub fn precompute_with_sync(
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        eats: &EATS,
        tool: &Tool,
        eps_e: f64,
        eps_b: f64,
        p_val: f64,
    ) -> Self {
        let ntheta = theta_data.len();
        let nt = t_data.len();
        let nvar = y_data.len();

        let all_dcos = compute_dcos_theta(theta_data);
        let bg_threshold = compute_bg_threshold(y_data, ntheta);
        let (phis, dphis) = build_phi_grid(theta_v, theta_data, y_data);
        let theta_groups = detect_theta_groups(y_data, ntheta, bg_threshold);

        let mut blasts_out = Vec::new();
        let mut t_obs_out = Vec::new();
        let mut domega_out = Vec::new();
        let mut sync_out = Vec::new();

        for j in 0..ntheta {
            let cell_max_bg = y_data[2][j].iter().copied().fold(0.0f64, f64::max);
            if cell_max_bg < bg_threshold {
                continue;
            }

            let time_indices = adaptive_time_indices(nt, y_data, j, t_data);
            let nt_sub = time_indices.len();

            let theta = theta_data[j];
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let rep = theta_groups[j];

            for (phi_idx, &phi) in phis.iter().enumerate() {
                let mu = cos_theta * theta_v.cos() + sin_theta * phi.cos() * theta_v.sin();

                let mut cell_blasts = Vec::with_capacity(nt_sub);
                let mut cell_tobs = Vec::with_capacity(nt_sub);
                let mut cell_sync = Vec::with_capacity(nt_sub);

                for &k in &time_indices {
                    let r = y_data[4][j][k];
                    cell_tobs.push(t_data[k] - r * mu / C_SPEED);

                    let val = [
                        y_data[0][rep][k],
                        y_data[1][rep][k],
                        y_data[2][rep][k],
                        y_data[3][rep][k],
                        y_data[4][rep][k],
                        y_data[5][rep][k],
                        if nvar > 6 { y_data[6][rep][k] } else { 0.0 },
                        t_data[k],
                    ];

                    let mut blast = Blast::default();
                    eats.derive_blast(theta, phi, theta_v, &val, tool, &mut blast);

                    let cached = CachedSyncParams::from_blast(&blast, eps_e, eps_b, p_val);
                    cell_sync.push(cached);
                    cell_blasts.push(blast);
                }

                blasts_out.push(cell_blasts);
                t_obs_out.push(cell_tobs);
                domega_out.push(all_dcos[j] * dphis[phi_idx]);
                sync_out.push(cell_sync);
            }
        }

        BlastGridCache {
            blasts: blasts_out,
            t_obs: t_obs_out,
            domega: domega_out,
            cached_sync: Some(sync_out),
        }
    }

    /// Build a ForwardGrid for a specific frequency from cached blast states.
    pub fn build_forward_grid(
        &self,
        nu_z: f64,
        param: &Dict,
        radiation_model: RadiationModel,
    ) -> ForwardGrid {
        let ncells = self.blasts.len();
        let mut lg2_t_obs = Vec::with_capacity(ncells);
        let mut lg2_dl_domega = Vec::with_capacity(ncells);
        let mut domega = Vec::with_capacity(ncells);
        let mut lg2_t_min = Vec::with_capacity(ncells);
        let mut lg2_t_max = Vec::with_capacity(ncells);

        for j in 0..ncells {
            let nt_sub = self.blasts[j].len();
            let mut cell_dl = Vec::with_capacity(nt_sub);

            for k in 0..nt_sub {
                let blast = &self.blasts[j][k];
                let nu_src = nu_z / blast.doppler;
                let intensity = radiation_model(nu_src, param, blast);
                cell_dl.push(
                    intensity
                        * blast.r * blast.r
                        * blast.doppler * blast.doppler * blast.doppler,
                );
            }

            let lg2_t = to_lg2(&self.t_obs[j]);
            let lg2_dl = to_lg2(&cell_dl);

            let t_min = *lg2_t.first().unwrap_or(&f64::NEG_INFINITY);
            let t_max = *lg2_t.last().unwrap_or(&f64::NEG_INFINITY);

            lg2_t_obs.push(lg2_t);
            lg2_dl_domega.push(lg2_dl);
            domega.push(self.domega[j]);
            lg2_t_min.push(t_min);
            lg2_t_max.push(t_max);
        }

        ForwardGrid {
            lg2_t_obs,
            lg2_dl_domega,
            domega,
            lg2_t_min,
            lg2_t_max,
        }
    }

    /// Build a ForwardGrid using cached sync params (fast path).
    /// Uses sync_from_cached/sync_smooth_from_cached instead of generic function pointer,
    /// skipping sync_params() recomputation.
    pub fn build_forward_grid_fast(
        &self,
        nu_z: f64,
        p_val: f64,
        model_name: &str,
    ) -> ForwardGrid {
        use crate::afterglow::models::{sync_from_cached, sync_smooth_from_cached, sync_dnp_from_cached};

        let cached_sync = self.cached_sync.as_ref()
            .expect("build_forward_grid_fast requires precompute_with_sync");

        let ncells = self.blasts.len();
        let mut lg2_t_obs = Vec::with_capacity(ncells);
        let mut lg2_dl_domega = Vec::with_capacity(ncells);
        let mut domega = Vec::with_capacity(ncells);
        let mut lg2_t_min = Vec::with_capacity(ncells);
        let mut lg2_t_max = Vec::with_capacity(ncells);

        // Select the fast evaluation function
        let eval_fn: fn(f64, f64, &CachedSyncParams) -> f64 = match model_name {
            "sync" => sync_from_cached,
            "sync_smooth" => sync_smooth_from_cached,
            "sync_dnp" => sync_dnp_from_cached,
            _ => sync_from_cached,
        };

        for j in 0..ncells {
            let nt_sub = self.blasts[j].len();
            let mut cell_dl = Vec::with_capacity(nt_sub);

            for k in 0..nt_sub {
                let blast = &self.blasts[j][k];
                let cached = &cached_sync[j][k];
                let nu_src = nu_z / blast.doppler;
                let intensity = eval_fn(nu_src, p_val, cached);
                cell_dl.push(
                    intensity
                        * blast.r * blast.r
                        * blast.doppler * blast.doppler * blast.doppler,
                );
            }

            let lg2_t = to_lg2(&self.t_obs[j]);
            let lg2_dl = to_lg2(&cell_dl);

            let t_min = *lg2_t.first().unwrap_or(&f64::NEG_INFINITY);
            let t_max = *lg2_t.last().unwrap_or(&f64::NEG_INFINITY);

            lg2_t_obs.push(lg2_t);
            lg2_dl_domega.push(lg2_dl);
            domega.push(self.domega[j]);
            lg2_t_min.push(t_min);
            lg2_t_max.push(t_max);
        }

        ForwardGrid {
            lg2_t_obs,
            lg2_dl_domega,
            domega,
            lg2_t_min,
            lg2_t_max,
        }
    }
}

impl ForwardGrid {
    /// Pre-compute the forward grid for a given frequency.
    /// Called once per unique nu before querying multiple observation times.
    pub fn precompute(
        nu_z: f64,
        theta_v: f64,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        eats: &EATS,
        tool: &Tool,
        param: &Dict,
        radiation_model: RadiationModel,
    ) -> Self {
        let ntheta = theta_data.len();
        let nt = t_data.len();
        let nvar = y_data.len();

        let all_dcos = compute_dcos_theta(theta_data);
        let bg_threshold = compute_bg_threshold(y_data, ntheta);
        let (phis, dphis) = build_phi_grid(theta_v, theta_data, y_data);
        let theta_groups = detect_theta_groups(y_data, ntheta, bg_threshold);

        let mut lg2_t_obs = Vec::new();
        let mut lg2_dl_domega = Vec::new();
        let mut domega = Vec::new();
        let mut lg2_t_min = Vec::new();
        let mut lg2_t_max = Vec::new();

        for j in 0..ntheta {
            let cell_max_bg = y_data[2][j].iter().copied().fold(0.0f64, f64::max);
            if cell_max_bg < bg_threshold {
                continue;
            }

            let time_indices = adaptive_time_indices(nt, y_data, j, t_data);
            let nt_sub = time_indices.len();

            let theta = theta_data[j];
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let rep = theta_groups[j];

            for (phi_idx, &phi) in phis.iter().enumerate() {
                let mu = cos_theta * theta_v.cos() + sin_theta * phi.cos() * theta_v.sin();

                let mut cell_tobs = Vec::with_capacity(nt_sub);
                let mut cell_dl = Vec::with_capacity(nt_sub);

                for &k in &time_indices {
                    let r = y_data[4][j][k];
                    cell_tobs.push(t_data[k] - r * mu / C_SPEED);

                    let val = [
                        y_data[0][rep][k],
                        y_data[1][rep][k],
                        y_data[2][rep][k],
                        y_data[3][rep][k],
                        y_data[4][rep][k],
                        y_data[5][rep][k],
                        if nvar > 6 { y_data[6][rep][k] } else { 0.0 },
                        t_data[k],
                    ];

                    let mut blast = Blast::default();
                    eats.derive_blast(theta, phi, theta_v, &val, tool, &mut blast);

                    let nu_src = nu_z / blast.doppler;
                    let intensity = radiation_model(nu_src, param, &blast);
                    cell_dl.push(
                        intensity
                            * blast.r * blast.r
                            * blast.doppler * blast.doppler * blast.doppler,
                    );
                }

                let lg2_t = to_lg2(&cell_tobs);
                let lg2_dl = to_lg2(&cell_dl);

                let t_min = *lg2_t.first().unwrap_or(&f64::NEG_INFINITY);
                let t_max = *lg2_t.last().unwrap_or(&f64::NEG_INFINITY);

                lg2_t_obs.push(lg2_t);
                lg2_dl_domega.push(lg2_dl);
                domega.push(all_dcos[j] * dphis[phi_idx]);
                lg2_t_min.push(t_min);
                lg2_t_max.push(t_max);
            }
        }

        ForwardGrid {
            lg2_t_obs,
            lg2_dl_domega,
            domega,
            lg2_t_min,
            lg2_t_max,
        }
    }

    /// Compute luminosity at a single de-redshifted observer time.
    pub fn luminosity(&self, tobs_z: f64) -> f64 {
        if tobs_z <= 0.0 {
            return 0.0;
        }
        let lg2_tq = tobs_z.log2();
        let ncells = self.lg2_t_obs.len();
        let mut total = 0.0;

        for j in 0..ncells {
            // Fast range check using pre-computed min/max
            if lg2_tq < self.lg2_t_min[j] || lg2_tq > self.lg2_t_max[j] {
                continue;
            }

            let t_arr = &self.lg2_t_obs[j];
            let dl_arr = &self.lg2_dl_domega[j];
            let nt = t_arr.len();

            // Binary search for bracketing index
            let mut lo = 0;
            let mut hi = nt - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if t_arr[mid] <= lg2_tq {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            // Log₂ interpolation: only 1 exp2 call per contributing cell
            if dl_arr[lo] != f64::NEG_INFINITY
                && dl_arr[hi] != f64::NEG_INFINITY
                && t_arr[hi] > t_arr[lo]
            {
                let frac = (lg2_tq - t_arr[lo]) / (t_arr[hi] - t_arr[lo]);
                let lg2_interp = dl_arr[lo] + frac * (dl_arr[hi] - dl_arr[lo]);
                total += lg2_interp.exp2() * self.domega[j];
            } else if dl_arr[lo] != f64::NEG_INFINITY {
                total += dl_arr[lo].exp2() * self.domega[j];
            }
        }

        // No trailing 2π — it's baked into domega
        total
    }

    /// Compute luminosity for a batch of sorted observer times.
    ///
    /// Uses a bracket-pointer scan: O(N_cells + N_queries) instead of
    /// O(N_cells × log(N_t)) for binary search per query.
    /// Each contributing cell requires only 1 exp2() call per query.
    pub fn luminosity_batch(&self, tobs_sorted: &[f64]) -> Vec<f64> {
        let nq = tobs_sorted.len();
        let ncells = self.lg2_t_obs.len();
        let mut result = vec![0.0f64; nq];

        if nq == 0 || ncells == 0 {
            return result;
        }

        // Pre-compute lg2 of all query times
        let lg2_queries: Vec<f64> = tobs_sorted
            .iter()
            .map(|&t| if t > 0.0 { t.log2() } else { f64::NEG_INFINITY })
            .collect();

        for j in 0..ncells {
            let t_arr = &self.lg2_t_obs[j];
            let dl_arr = &self.lg2_dl_domega[j];
            let nt = t_arr.len();
            let domega_j = self.domega[j];

            if nt < 2 {
                continue;
            }

            let t_min = self.lg2_t_min[j];
            let t_max = self.lg2_t_max[j];

            // Find first query in range via binary search
            let q_start = match lg2_queries.binary_search_by(|q| q.partial_cmp(&t_min).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(i) => i,
                Err(i) => i,
            };

            if q_start >= nq {
                continue;
            }

            // Bracket pointer: advance through the cell's time array
            let mut lo = 0;

            for qi in q_start..nq {
                let lg2_tq = lg2_queries[qi];

                // Past this cell's range — no more queries will hit
                if lg2_tq > t_max {
                    break;
                }

                // Before this cell's range
                if lg2_tq < t_min {
                    continue;
                }

                // Advance bracket pointer
                while lo + 1 < nt && t_arr[lo + 1] <= lg2_tq {
                    lo += 1;
                }

                let hi = if lo + 1 < nt { lo + 1 } else { lo };

                // Log₂ interpolation: 1 exp2 per contributing cell per query
                if dl_arr[lo] != f64::NEG_INFINITY
                    && dl_arr[hi] != f64::NEG_INFINITY
                    && t_arr[hi] > t_arr[lo]
                {
                    let frac = (lg2_tq - t_arr[lo]) / (t_arr[hi] - t_arr[lo]);
                    let lg2_interp = dl_arr[lo] + frac * (dl_arr[hi] - dl_arr[lo]);
                    result[qi] += lg2_interp.exp2() * domega_j;
                } else if dl_arr[lo] != f64::NEG_INFINITY {
                    result[qi] += dl_arr[lo].exp2() * domega_j;
                }
            }
        }

        result
    }
}
