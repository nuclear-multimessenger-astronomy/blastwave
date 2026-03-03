use crate::constants::*;
use crate::hydro::tools::Tool;
use crate::afterglow::blast::Blast;
use crate::afterglow::eats::EATS;
use crate::afterglow::models::{Dict, RadiationModel};

/// Pre-computed forward-mapping grid for fast on-axis luminosity computation.
///
/// Instead of EATS inverse-mapping (root-finding per observation point), this
/// pre-computes observer-frame times and radiation at each hydro grid point,
/// then uses binary search + interpolation for each query.
pub struct ForwardGrid {
    /// Observer-frame times (de-redshifted): [ntheta][nt]
    t_obs: Vec<Vec<f64>>,
    /// Pre-computed dL/dOmega: [ntheta][nt]
    dl_domega: Vec<Vec<f64>>,
    /// Solid angle weights: cos(theta_{j-1/2}) - cos(theta_{j+1/2})
    dcos_theta: Vec<f64>,
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

        // Compute solid angle weights from cell boundaries.
        // boundary[0] = 0, boundary[j] = midpoint of adjacent centers, boundary[ntheta] = pi
        let mut boundaries = vec![0.0; ntheta + 1];
        boundaries[0] = 0.0;
        for j in 1..ntheta {
            boundaries[j] = (theta_data[j - 1] + theta_data[j]) / 2.0;
        }
        boundaries[ntheta] = PI;

        let all_dcos: Vec<f64> = (0..ntheta)
            .map(|j| boundaries[j].cos() - boundaries[j + 1].cos())
            .collect();

        // Skip cells with negligible energy (artificial tail).
        // Use peak beta_gamma_sq across all time steps (handles spreading).
        let max_bg_sq = (0..ntheta)
            .flat_map(|j| y_data[2][j].iter().copied())
            .fold(0.0f64, f64::max);
        let bg_sq_threshold = max_bg_sq * 1e-6;

        // Subsample time grid when it exceeds max_nt to limit pre-computation cost.
        // PDE mode outputs ~1000 time steps; ODE/no-spread output ~150.
        const MAX_NT: usize = 300;
        let time_indices: Vec<usize> = if nt <= MAX_NT {
            (0..nt).collect()
        } else {
            let stride = nt / MAX_NT;
            let mut indices: Vec<usize> = (0..nt).step_by(stride).collect();
            // Always include the last index for correct boundary coverage
            if *indices.last().unwrap() != nt - 1 {
                indices.push(nt - 1);
            }
            indices
        };
        let nt_sub = time_indices.len();

        let mut t_obs = Vec::new();
        let mut dl_domega = Vec::new();
        let mut dcos_theta = Vec::new();

        for j in 0..ntheta {
            let cell_max_bg = y_data[2][j].iter().copied().fold(0.0f64, f64::max);
            if cell_max_bg < bg_sq_threshold {
                continue;
            }

            let cos_theta = theta_data[j].cos();
            let mut cell_tobs = Vec::with_capacity(nt_sub);
            let mut cell_dl = Vec::with_capacity(nt_sub);

            for &k in &time_indices {
                let r = y_data[4][j][k];
                cell_tobs.push(t_data[k] - r * cos_theta / C_SPEED);

                let val = [
                    y_data[0][j][k],
                    y_data[1][j][k],
                    y_data[2][j][k],
                    y_data[3][j][k],
                    y_data[4][j][k],
                    y_data[5][j][k],
                    if y_data.len() > 6 { y_data[6][j][k] } else { 0.0 }, // t_comv
                    t_data[k],
                ];

                let mut blast = Blast::default();
                eats.derive_blast(theta_data[j], 0.0, theta_v, &val, tool, &mut blast);

                let nu_src = nu_z / blast.doppler;
                let intensity = radiation_model(nu_src, param, &blast);
                cell_dl.push(
                    intensity
                        * blast.r * blast.r
                        * blast.doppler * blast.doppler * blast.doppler,
                );
            }

            t_obs.push(cell_tobs);
            dl_domega.push(cell_dl);
            dcos_theta.push(all_dcos[j]);
        }

        ForwardGrid {
            t_obs,
            dl_domega,
            dcos_theta,
        }
    }

    /// Compute luminosity at a single de-redshifted observer time.
    pub fn luminosity(&self, tobs_z: f64) -> f64 {
        let ncells = self.t_obs.len();
        let mut total = 0.0;

        for j in 0..ncells {
            let t_arr = &self.t_obs[j];
            let dl_arr = &self.dl_domega[j];
            let nt = t_arr.len();

            // Skip if outside the time range for this cell
            if tobs_z < t_arr[0] || tobs_z > t_arr[nt - 1] {
                continue;
            }

            // Binary search for bracketing index
            let mut lo = 0;
            let mut hi = nt - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if t_arr[mid] <= tobs_z {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            // Log-linear interpolation (log in both t and dl)
            let dl_interp = if dl_arr[lo] > 0.0
                && dl_arr[hi] > 0.0
                && t_arr[lo] > 0.0
                && t_arr[hi] > t_arr[lo]
            {
                let log_frac = (tobs_z.ln() - t_arr[lo].ln())
                    / (t_arr[hi].ln() - t_arr[lo].ln());
                (dl_arr[lo].ln() + log_frac * (dl_arr[hi].ln() - dl_arr[lo].ln())).exp()
            } else if t_arr[hi] > t_arr[lo] {
                // Linear fallback for zero/negative values
                let frac = (tobs_z - t_arr[lo]) / (t_arr[hi] - t_arr[lo]);
                dl_arr[lo] + frac * (dl_arr[hi] - dl_arr[lo])
            } else {
                dl_arr[lo]
            };

            total += dl_interp.max(0.0) * self.dcos_theta[j];
        }

        total * 2.0 * PI
    }
}
