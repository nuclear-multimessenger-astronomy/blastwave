use crate::constants::*;
use crate::hydro::config::{JetConfig, SpreadMode};
use crate::hydro::tools::Tool;
use crate::hydro::reverse_shock::{self, FRShockEqn, NVAR_RS};
use rayon::prelude::*;

const C2: f64 = C_SPEED * C_SPEED;

/// PDE solver for jet hydrodynamics.
pub struct SimBox {
    // tools
    tool: Tool,

    // configuration
    tmin: f64,
    tmax: f64,
    cfl: f64,
    spread: bool,
    spread_mode: SpreadMode,
    theta_c: f64,

    // mesh
    ntheta: usize,
    theta: Vec<f64>,
    theta_edge: Vec<f64>,

    // conserved variables
    eb: Vec<f64>,
    ht: Vec<f64>,
    msw: Vec<f64>,
    mej: Vec<f64>,
    r: Vec<f64>,
    u2_th: Vec<f64>,

    // primitive variables
    beta_gamma_sq: Vec<f64>,
    beta_th: Vec<f64>,
    psw: Vec<f64>,
    hb: Vec<f64>,
    s: Vec<f64>,

    // convenient variables
    beta: Vec<f64>,
    gamma: Vec<f64>,

    // eigenvalues
    eigenvalues: Vec<f64>,
    alpha_r: Vec<f64>,

    // slope: [5][ntheta]
    slope: Vec<Vec<f64>>,
    r_slope_l: Vec<f64>,
    r_slope_r: Vec<f64>,

    // numerical flux: [4][ntheta+1]
    numerical_flux: Vec<Vec<f64>>,
    dr_dt: Vec<f64>,

    // dy/dt: [5][ntheta]
    dy_dt: Vec<Vec<f64>>,

    // Pre-allocated scratch buffer for RK2 (avoids per-step heap allocation)
    rk2_scratch: Vec<Vec<f64>>,

    // PDE solution: [5][ntheta][nt]
    pub ys: Vec<Vec<Vec<f64>>>,
    pub ts: Vec<f64>,

    // Reverse shock solution: [NVAR_RS][ntheta][nt]
    // Only populated when include_reverse_shock is true
    pub ys_rs: Option<Vec<Vec<Vec<f64>>>>,
    pub crossing_idx: Vec<usize>, // per-theta crossing time index

    // Config reference for RS
    include_reverse_shock: bool,
    config_rs: Option<JetConfig>,

    // Forward shock microphysics
    eps_e: f64,  // electron energy fraction (affects thermal energy budget)
    eps_b: f64,  // magnetic energy fraction (for eps_rad computation)
    p_fwd: f64,  // electron spectral index (for eps_rad computation)
    // Pre-computed eps_rad coefficients (matching VegasAfterglow)
    gamma_m_coeff: f64, // (p-2)/(p-1) * eps_e * mp/me
    gamma_c_coeff: f64, // 6π * me * c / (σ_T * eps_b)

    // Forward shock energy injection (magnetar spin-down, multiple episodes)
    // Each element: (l0_per_sr_c2, t0, q, ts)
    magnetar_episodes: Vec<(f64, f64, f64, f64)>,
}

impl SimBox {
    pub fn new(config: &JetConfig) -> Self {
        let tool = Tool::new_with_k(config.nwind, config.nism, config.k, config.rtol, config.cal_level);
        let ntheta = config.eb.len();

        let mut theta = vec![0.0; ntheta];
        for i in 0..ntheta {
            theta[i] = (config.theta_edge[i] + config.theta_edge[i + 1]) / 2.0;
        }

        let mut sb = SimBox {
            tool,
            tmin: config.tmin,
            tmax: config.tmax,
            cfl: config.cfl,
            spread: config.spread,
            spread_mode: config.spread_mode,
            theta_c: config.theta_c,
            ntheta,
            theta,
            theta_edge: config.theta_edge.clone(),
            eb: config.eb.clone(),
            ht: config.ht.clone(),
            msw: config.msw.clone(),
            mej: config.mej.clone(),
            r: config.r.clone(),
            u2_th: vec![0.0; ntheta],
            beta_gamma_sq: vec![0.0; ntheta],
            beta_th: vec![0.0; ntheta],
            psw: vec![0.0; ntheta],
            hb: vec![0.0; ntheta],
            s: vec![0.0; ntheta],
            beta: vec![0.0; ntheta],
            gamma: vec![0.0; ntheta],
            eigenvalues: vec![0.0; ntheta],
            alpha_r: vec![0.0; ntheta],
            slope: vec![vec![0.0; ntheta]; 6],
            r_slope_l: vec![0.0; ntheta],
            r_slope_r: vec![0.0; ntheta],
            numerical_flux: vec![vec![0.0; ntheta + 1]; 5],
            dr_dt: vec![0.0; ntheta],
            dy_dt: vec![vec![0.0; ntheta]; 6],
            rk2_scratch: vec![vec![0.0; ntheta]; 6],
            ys: Vec::new(),
            ts: Vec::new(),
            ys_rs: None,
            crossing_idx: Vec::new(),
            include_reverse_shock: config.include_reverse_shock,
            config_rs: if config.include_reverse_shock { Some(config.clone()) } else { None },
            eps_e: config.eps_e,
            eps_b: config.eps_b,
            p_fwd: config.p_fwd,
            gamma_m_coeff: if config.eps_e > 0.0 && config.p_fwd > 2.0 {
                (config.p_fwd - 2.0) / (config.p_fwd - 1.0) * config.eps_e * MASS_P / MASS_E
            } else { 0.0 },
            gamma_c_coeff: if config.eps_b > 0.0 {
                6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * config.eps_b)
            } else { 0.0 },
            magnetar_episodes: config.magnetar_l0.iter().enumerate().map(|(i, &l0)| {
                let t0 = config.magnetar_t0.get(i).copied().unwrap_or(1.0);
                let q = config.magnetar_q.get(i).copied().unwrap_or(2.0);
                let ts = config.magnetar_ts.get(i).copied().unwrap_or(0.0);
                (l0 / (4.0 * PI * C_SPEED * C_SPEED), t0, q, ts)
            }).filter(|&(l0_c2, _, _, _)| l0_c2 > 0.0).collect(),
        };

        sb.solve_primitive();
        sb.solve_eigen();
        let one_minus_eps_e = 1.0 - sb.eps_e;
        for i in 0..ntheta {
            sb.u2_th[i] = one_minus_eps_e * (sb.gamma[i] - 1.0) * sb.msw[i] * C2;
        }
        sb
    }

    pub fn get_theta(&self) -> &Vec<f64> {
        &self.theta
    }

    pub fn tool(&self) -> &Tool {
        &self.tool
    }

    pub fn solve_pde(&mut self) {
        if !self.magnetar_episodes.is_empty() && self.spread_mode == SpreadMode::Pde {
            eprintln!("WARNING: magnetar injection not supported in PDE mode; use spread_mode=\"ode\" or \"none\"");
        }

        match self.spread_mode {
            SpreadMode::None => self.solve_no_spread(),
            SpreadMode::Ode => self.solve_ode_spread(),
            SpreadMode::Pde => self.solve_spread(),
        }

        // Solve reverse shock if enabled
        if self.include_reverse_shock {
            self.solve_reverse_shock();
            self.apply_rs_energy_correction();
        }
    }

    /// Solve the reverse shock ODE for each theta cell using the forward shock
    /// PDE solution as the time grid.
    pub fn solve_reverse_shock(&mut self) {
        let config = self.config_rs.as_ref().unwrap().clone();
        let ntheta = self.ntheta;
        let nt = self.ts.len();
        let tmin = self.ts[0];
        let tmax = *self.ts.last().unwrap();

        // Parallel RS solve over theta cells
        let ys_ref = &self.ys;
        let ts_ref = &self.ts;
        let cell_results: Vec<(Vec<Vec<f64>>, usize)> = (0..ntheta).into_par_iter()
            .map(|j| {
                let mej = ys_ref[1][j][0];
                let bg_sq = ys_ref[2][j][0];
                let gamma4 = (bg_sq + 1.0).sqrt();

                if gamma4 < 1.5 {
                    return (vec![vec![0.0; nt]; NVAR_RS], nt);
                }

                let eps4_init = (gamma4 - 1.0) * mej * C_SPEED * C_SPEED + mej * C_SPEED * C_SPEED;

                let mut eqn = FRShockEqn::new(
                    config.nwind,
                    config.nism,
                    config.k,
                    gamma4,
                    mej,
                    eps4_init,
                    config.sigma,
                    0.1,
                    0.01,
                    2.3,
                    config.eps_e_rs,
                    config.eps_b_rs,
                    config.p_rs,
                    config.duration,
                    config.t0_injection,
                    config.l_injection,
                    config.m_dot_injection,
                    config.rtol,
                );

                let (cell_rs, cross_idx) = reverse_shock::solve_reverse_shock_cell(
                    &mut eqn,
                    tmin,
                    tmax,
                    config.rtol.max(1e-4),
                    ts_ref,
                );

                (cell_rs, cross_idx)
            })
            .collect();

        // Unpack parallel results into [NVAR_RS][ntheta][nt] layout
        let mut rs_data: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); ntheta]; NVAR_RS];
        let mut crossing_indices = vec![nt; ntheta];
        for (j, (cell_rs, cross_idx)) in cell_results.into_iter().enumerate() {
            crossing_indices[j] = cross_idx;
            for var in 0..NVAR_RS {
                rs_data[var][j] = cell_rs[var].clone();
            }
        }

        // Post-process: fill rs_data[13] with the crossing lab time for each theta cell.
        // This lets the EATS solver know when the RS finished crossing the ejecta.
        // Use a very large but finite sentinel (1e30) for cells that never cross,
        // so that t > t_crossing is never satisfied and RS data stays finite.
        for j in 0..ntheta {
            let cidx = crossing_indices[j];
            let t_cross = if cidx < nt {
                self.ts[cidx]
            } else {
                1e30 // sentinel: no crossing
            };
            for k in 0..rs_data[13][j].len() {
                rs_data[13][j][k] = t_cross;
            }
        }

        self.ys_rs = Some(rs_data);
        self.crossing_idx = crossing_indices;
    }

    /// Post-processing correction: subtract RS thermal energy from the FS
    /// blast energy and re-solve for the corrected beta_gamma_sq.
    ///
    /// The PDE doesn't account for energy drained by the reverse shock, so
    /// FS Gamma is too high during the crossing phase. This corrects by
    /// saying "FS energy = total blast energy − RS thermal energy" and
    /// re-solving the energy equation for the new Gamma.
    fn apply_rs_energy_correction(&mut self) {
        let rs_data = self.ys_rs.as_ref().unwrap();
        let nt = self.ts.len();
        let c2 = C_SPEED * C_SPEED;

        for j in 0..self.ntheta {
            for k in 0..nt {
                let u3_th = rs_data[4][j][k]; // RS thermal energy per sr [erg/sr]
                if u3_th <= 0.0 {
                    continue;
                }

                // Convert u3_th from erg/sr to g/sr (same units as eb)
                let u3_th_mass = u3_th / c2;

                // Reconstruct eb from stored primitives
                let msw = self.ys[0][j][k];
                let mej = self.ys[1][j][k];
                let bg2 = self.ys[2][j][k];
                let r = self.ys[4][j][k];

                let s = self.tool.solve_s(r, bg2);
                let gamma = (bg2 + 1.0).sqrt();
                let beta_sq = bg2 / (bg2 + 1.0);
                let eb = s * gamma * gamma * (1.0 + beta_sq * beta_sq / 3.0) * msw
                    + gamma * ((1.0 - s) * msw + mej);

                // Subtract RS thermal energy
                let eb_corr = eb - u3_th_mass;
                if eb_corr <= mej + msw {
                    // Corrected energy must exceed rest mass
                    continue;
                }

                // Re-solve for beta_gamma_sq with corrected energy
                if let Ok(bg2_new) =
                    self.tool
                        .solve_beta_gamma_sq(msw / eb_corr, mej / eb_corr, r)
                {
                    if bg2_new < bg2 && bg2_new > 0.0 {
                        self.ys[2][j][k] = bg2_new;
                    }
                }
            }
        }
    }

    fn solve_primitive(&mut self) {
        for i in 0..self.ntheta {
            self.beta_gamma_sq[i] = self
                .tool
                .solve_beta_gamma_sq(self.msw[i] / self.eb[i], self.mej[i] / self.eb[i], self.r[i])
                .unwrap_or_else(|e| panic!("Hydro Primitive solver: {}", e));

            self.s[i] = self.tool.solve_s(self.r[i], self.beta_gamma_sq[i]);
            self.gamma[i] = (self.beta_gamma_sq[i] + 1.0).sqrt();
            self.beta[i] = (self.beta_gamma_sq[i] / (self.beta_gamma_sq[i] + 1.0)).sqrt();
            self.psw[i] = self.s[i] * self.beta[i] * self.beta[i] * self.msw[i] / 3.0;
            self.hb[i] = self.eb[i] + self.psw[i];
            self.beta_th[i] = self.ht[i] / self.hb[i];
        }
    }

    fn solve_eigen(&mut self) {
        for i in 0..self.ntheta {
            let g = self.gamma[i];
            let b = self.beta[i];
            let a_val = 2.0 * self.s[i] / 3.0 * self.msw[i] * (4.0 * g * g * g * g - 1.0)
                + ((1.0 - self.s[i]) * self.msw[i] + self.mej[i]) * g * g * g;
            let d_psw_d_eb = 2.0 * self.s[i] / 3.0 * self.msw[i] / a_val;
            let d_psw_d_msw = self.s[i] * b * b / 3.0
                - 2.0 * self.s[i] / 3.0 * (self.eb[i] - g * self.mej[i]) / a_val;
            let d_psw_d_mej = -2.0 * self.s[i] / 3.0 * g * self.msw[i] / a_val;

            let b_val =
                self.mej[i] / self.hb[i] * d_psw_d_mej + self.msw[i] / self.hb[i] * d_psw_d_msw;
            let bt = self.beta_th[i];
            let c_val = ((1.0 - bt * bt) * (d_psw_d_eb + b_val) + bt * bt / 4.0 * b_val * b_val)
                .sqrt();
            let alpha1 = bt.abs();
            let alpha2 = (bt * (1.0 - b_val / 2.0) + c_val).abs();
            let alpha3 = (bt * (1.0 - b_val / 2.0) - c_val).abs();

            self.eigenvalues[i] = alpha1.max(alpha2).max(alpha3) * C_SPEED / self.r[i];
            self.alpha_r[i] = bt.abs() / self.r[i];
        }
    }

    fn solve_slope(&mut self) {
        // For msw, mej, beta_gamma_sq, beta_th, u2_th (indices 0..5)
        for j in 0..5usize {
            for i in 0..self.ntheta {
                let index1 = if i > 0 { i - 1 } else { 0 };
                let index2 = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };

                let var = match j {
                    0 => &self.msw,
                    1 => &self.mej,
                    2 => &self.beta_gamma_sq,
                    3 => &self.beta_th,
                    4 => &self.u2_th,
                    _ => unreachable!(),
                };

                let diff1 = var[i] - var[index1];
                let diff2 = var[index2] - var[i];
                let slope1 = if i == index1 {
                    0.0
                } else {
                    diff1 / (self.theta[i] - self.theta[index1])
                };
                let slope2 = if i == index2 {
                    0.0
                } else {
                    diff2 / (self.theta[index2] - self.theta[i])
                };
                self.slope[j][i] = self.tool.minmod(slope1, slope2);
            }
        }

        // R slopes
        for i in 0..self.ntheta {
            let index1 = if i > 0 { i - 1 } else { 0 };
            let index2 = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };
            let diff1 = self.r[i] - self.r[index1];
            let diff2 = self.r[index2] - self.r[i];
            self.r_slope_l[i] = if i == index1 {
                0.0
            } else {
                diff1 / (self.theta[i] - self.theta[index1])
            };
            self.r_slope_r[i] = if i == index2 {
                0.0
            } else {
                diff2 / (self.theta[index2] - self.theta[i])
            };
            self.slope[5][i] = self.tool.minmod(self.r_slope_l[i], self.r_slope_r[i]);
        }
    }

    fn solve_numerical_flux(&mut self) {
        for i in 1..self.ntheta {
            // Reconstruct primitive variables at face i
            // slope indices: 0=msw, 1=mej, 2=bg_sq, 3=beta_th, 4=u2_th, 5=r
            let vars: [&Vec<f64>; 6] = [
                &self.msw,
                &self.mej,
                &self.beta_gamma_sq,
                &self.beta_th,
                &self.u2_th,
                &self.r,
            ];

            let mut var_l = [0.0f64; 6];
            let mut var_r = [0.0f64; 6];

            for j in 0..6 {
                var_l[j] =
                    vars[j][i - 1] + self.slope[j][i - 1] * (self.theta_edge[i] - self.theta[i - 1]);
                var_r[j] = vars[j][i] + self.slope[j][i] * (self.theta_edge[i] - self.theta[i]);
            }

            // Aliases
            let msw_l = var_l[0];
            let mej_l = var_l[1];
            let bg_sq_l = var_l[2];
            let bt_l = var_l[3];
            let u2th_l = var_l[4];
            let r_l = var_l[5];

            let msw_r = var_r[0];
            let mej_r = var_r[1];
            let bg_sq_r = var_r[2];
            let bt_r = var_r[3];
            let u2th_r = var_r[4];
            let r_r = var_r[5];

            let s_l = self.tool.solve_s(r_l, bg_sq_l);
            let s_r = self.tool.solve_s(r_r, bg_sq_r);

            // Left-biased conserved
            let eb_l = s_l
                * (1.0 + bg_sq_l * bg_sq_l / (bg_sq_l + 1.0) / (bg_sq_l + 1.0) / 3.0)
                * (bg_sq_l + 1.0)
                * msw_l
                + (1.0 - s_l) * (bg_sq_l + 1.0).sqrt() * msw_l
                + (bg_sq_l + 1.0).sqrt() * mej_l;
            let psw_l = s_l * bg_sq_l / (bg_sq_l + 1.0) * msw_l / 3.0;
            let ht_l = (eb_l + psw_l) * bt_l;

            // Right-biased conserved
            let eb_r = s_r
                * (1.0 + bg_sq_r * bg_sq_r / (bg_sq_r + 1.0) / (bg_sq_r + 1.0) / 3.0)
                * (bg_sq_r + 1.0)
                * msw_r
                + (1.0 - s_r) * (bg_sq_r + 1.0).sqrt() * msw_r
                + (bg_sq_r + 1.0).sqrt() * mej_r;
            let psw_r = s_r * bg_sq_r / (bg_sq_r + 1.0) * msw_r / 3.0;
            let ht_r = (eb_r + psw_r) * bt_r;

            // Physical flux (indices 0-3: eb, ht, msw, mej; 4: u2_th)
            let fl = [
                ht_l / r_l * C_SPEED,
                (ht_l * bt_l + psw_l) / r_l * C_SPEED,
                msw_l * bt_l / r_l * C_SPEED,
                mej_l * bt_l / r_l * C_SPEED,
                u2th_l * bt_l / r_l * C_SPEED,
            ];
            let fr = [
                ht_r / r_r * C_SPEED,
                (ht_r * bt_r + psw_r) / r_r * C_SPEED,
                msw_r * bt_r / r_r * C_SPEED,
                mej_r * bt_r / r_r * C_SPEED,
                u2th_r * bt_r / r_r * C_SPEED,
            ];

            // Viscosity
            let il = if i >= 2 { i - 2 } else { 0 };
            let ir = if i + 1 < self.ntheta { i + 1 } else { self.ntheta - 1 };
            let alpha = self.eigenvalues[il..ir]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let sin_edge = self.theta_edge[i].sin();
            self.numerical_flux[0][i] =
                0.5 * (fl[0] + fr[0] - alpha * (eb_r - eb_l)) * sin_edge;
            self.numerical_flux[1][i] =
                0.5 * (fl[1] + fr[1] - alpha * (ht_r - ht_l)) * sin_edge;
            self.numerical_flux[2][i] =
                0.5 * (fl[2] + fr[2] - alpha * (msw_r - msw_l)) * sin_edge;
            self.numerical_flux[3][i] =
                0.5 * (fl[3] + fr[3] - alpha * (mej_r - mej_l)) * sin_edge;
            self.numerical_flux[4][i] =
                0.5 * (fl[4] + fr[4] - alpha * (u2th_r - u2th_l)) * sin_edge;
        }
    }

    fn solve_delta_t(&self) -> f64 {
        let mut delta_t_min = f64::MAX;
        for i in 0..self.ntheta {
            let omega = self.beta[i] * C_SPEED / self.r[i];
            let omega_all = self.eigenvalues[i] + 0.05 * omega;
            let delta_t = self.cfl * (self.theta_edge[i + 1] - self.theta_edge[i]) / omega_all;
            delta_t_min = delta_t_min.min(delta_t);
        }
        delta_t_min
    }

    fn solve_dy_dt(&mut self) {
        for i in 0..self.ntheta {
            let beta_f = 4.0 * self.beta[i] * (self.beta_gamma_sq[i] + 1.0)
                / (4.0 * self.beta_gamma_sq[i] + 3.0);
            let beta_r = if self.beta_th[i] <= self.beta[i] {
                (self.beta[i] * self.beta[i] - self.beta_th[i] * self.beta_th[i]).sqrt()
            } else {
                0.0
            };
            let vol =
                self.theta_edge[i].cos() - self.theta_edge[i + 1].cos();

            // dR/dt (R slope is at index 5)
            let il = if i > 0 { i - 1 } else { 0 };
            let ir = if i < self.ntheta - 1 { i + 1 } else { self.ntheta - 1 };
            let alpha = self.alpha_r[il].max(self.alpha_r[i]).max(self.alpha_r[ir]);
            self.dy_dt[4][i] = (beta_f - self.slope[5][i] * self.beta_th[i] / self.r[i]
                + 0.5 * alpha * (self.r_slope_r[i] - self.r_slope_l[i]))
                * C_SPEED;

            let rho = self.tool.solve_density(self.r[i]) * MASS_P;
            let dmsw_dt = self.dy_dt[4][i] * rho * self.r[i] * self.r[i];
            self.dy_dt[0][i] = (self.numerical_flux[0][i] - self.numerical_flux[0][i + 1]) / vol
                + dmsw_dt;
            self.dy_dt[1][i] = (self.numerical_flux[1][i] - self.numerical_flux[1][i + 1]) / vol
                + (self.theta[i].cos() / self.theta[i].sin() * self.psw[i]
                    - self.ht[i] * beta_r)
                    * C_SPEED
                    / self.r[i];
            self.dy_dt[2][i] = (self.numerical_flux[2][i] - self.numerical_flux[2][i + 1]) / vol
                + dmsw_dt;
            self.dy_dt[3][i] = (self.numerical_flux[3][i] - self.numerical_flux[3][i + 1]) / vol;

            // u2_th evolution (simplified PDE: omit dGamma/dt term)
            let ad_idx = reverse_shock::adiabatic_idx(self.gamma[i]);
            let dln_v_dt = 3.0 * self.dy_dt[4][i] / self.r[i];
            let heating = (self.gamma[i] - 1.0) * C2 * dmsw_dt;
            let cooling = -(ad_idx - 1.0) * dln_v_dt * self.u2_th[i];
            self.dy_dt[5][i] = (self.numerical_flux[4][i] - self.numerical_flux[4][i + 1]) / vol
                + heating + cooling;
        }
    }

    fn one_step_rk2(&mut self, dt: f64) {
        // Save initial state into pre-allocated scratch (no heap allocation)
        for j in 0..self.ntheta {
            self.rk2_scratch[0][j] = self.eb[j];
            self.rk2_scratch[1][j] = self.ht[j];
            self.rk2_scratch[2][j] = self.msw[j];
            self.rk2_scratch[3][j] = self.mej[j];
            self.rk2_scratch[4][j] = self.r[j];
            self.rk2_scratch[5][j] = self.u2_th[j];
        }

        // Step 1
        self.solve_slope();
        self.solve_numerical_flux();
        self.solve_dy_dt();

        for j in 0..self.ntheta {
            self.eb[j] += dt * self.dy_dt[0][j];
            self.ht[j] += dt * self.dy_dt[1][j];
            self.msw[j] += dt * self.dy_dt[2][j];
            self.mej[j] += dt * self.dy_dt[3][j];
            self.r[j] += dt * self.dy_dt[4][j];
            self.u2_th[j] += dt * self.dy_dt[5][j];
            self.u2_th[j] = self.u2_th[j].max(0.0);
        }
        self.solve_primitive();
        self.solve_eigen();

        // Step 2
        self.solve_slope();
        self.solve_numerical_flux();
        self.solve_dy_dt();

        for j in 0..self.ntheta {
            self.eb[j] = 0.5 * self.rk2_scratch[0][j] + 0.5 * self.eb[j] + 0.5 * dt * self.dy_dt[0][j];
            self.ht[j] = 0.5 * self.rk2_scratch[1][j] + 0.5 * self.ht[j] + 0.5 * dt * self.dy_dt[1][j];
            self.msw[j] = 0.5 * self.rk2_scratch[2][j] + 0.5 * self.msw[j] + 0.5 * dt * self.dy_dt[2][j];
            self.mej[j] = 0.5 * self.rk2_scratch[3][j] + 0.5 * self.mej[j] + 0.5 * dt * self.dy_dt[3][j];
            self.r[j] = 0.5 * self.rk2_scratch[4][j] + 0.5 * self.r[j] + 0.5 * dt * self.dy_dt[4][j];
            self.u2_th[j] = 0.5 * self.rk2_scratch[5][j] + 0.5 * self.u2_th[j] + 0.5 * dt * self.dy_dt[5][j];
            self.u2_th[j] = self.u2_th[j].max(0.0);
        }
        self.solve_primitive();
        self.solve_eigen();
    }

    fn save_primitives(&mut self) {
        let primitives: [&Vec<f64>; 6] = [
            &self.msw,
            &self.mej,
            &self.beta_gamma_sq,
            &self.beta_th,
            &self.r,
            &self.u2_th,
        ];
        for i in 0..6 {
            for j in 0..self.ntheta {
                self.ys[i][j].push(primitives[i][j]);
            }
        }
    }

    fn init_solution(&mut self) {
        self.ts.push(self.tmin);
        self.ys = vec![vec![Vec::new(); self.ntheta]; 6];
        self.save_primitives();
    }

    /// Maximum output time points for the CFL-limited PDE solver.
    /// Since PDE step count is determined by CFL (not output schedule),
    /// more output points don't slow it down but do improve EATS interpolation.
    const PDE_OUTPUT_POINTS: usize = 1000;

    /// Maximum output time points for adaptive (ODE/no-spread) solvers.
    /// These solvers' step size is constrained by the output schedule, so
    /// fewer points = fewer forced steps = faster. VegasAfterglow uses ~94.
    const ADAPTIVE_OUTPUT_POINTS: usize = 150;

    fn solve_spread(&mut self) {
        self.init_solution();

        // Pre-compute log-spaced output schedule to avoid saving every CFL step
        let log_tmin = self.tmin.ln();
        let log_tmax = self.tmax.ln();
        let d_log_t = (log_tmax - log_tmin) / Self::PDE_OUTPUT_POINTS as f64;
        let mut next_save_log_t = log_tmin + d_log_t;

        let mut t = self.tmin;
        while t < self.tmax {
            let dt = self.solve_delta_t().min(self.tmax - t + 1e-6);
            self.one_step_rk2(dt);
            t += dt;

            // Only save at log-spaced intervals (or at the final step)
            if t.ln() >= next_save_log_t || t >= self.tmax {
                self.ts.push(t);
                self.save_primitives();
                // Advance past all crossed output times
                while next_save_log_t <= t.ln() {
                    next_save_log_t += d_log_t;
                }
            }
        }
    }

    /// Count contiguous cells from index 0 that share identical initial conditions.
    /// For a tophat jet with a tail, returns the number of core cells (theta < theta_c)
    /// that all have the same energy/mass/radius. Tail cells beyond theta_c differ.
    fn count_uniform_core(&self) -> usize {
        if self.ntheta <= 1 {
            return self.ntheta;
        }
        let mut n = 1;
        for i in 1..self.ntheta {
            if (self.eb[i] - self.eb[0]).abs() > self.eb[0].abs() * 1e-12
                || (self.msw[i] - self.msw[0]).abs() > self.msw[0].abs() * 1e-12
                || (self.mej[i] - self.mej[0]).abs() > self.mej[0].abs().max(1e-30) * 1e-12
                || (self.r[i] - self.r[0]).abs() > self.r[0].abs() * 1e-12
            {
                break;
            }
            n += 1;
        }
        n
    }

    /// Compute radiative efficiency (matching VegasAfterglow's formula).
    /// Returns eps_rad ∈ [0, eps_e].
    fn compute_eps_rad(&self, gamma: f64, t_comv: f64, rho: f64) -> f64 {
        if self.eps_e <= 0.0 || self.eps_b <= 0.0 || self.p_fwd <= 2.0 || t_comv <= 0.0 {
            return 0.0;
        }
        let e_th = (gamma - 1.0) * 4.0 * gamma * rho * C2;
        if e_th <= 0.0 { return 0.0; }
        let gm = self.gamma_m_coeff * (gamma - 1.0) + 1.0;
        let gc = (self.gamma_c_coeff / (e_th * t_comv)).max(1.0);
        let ratio = gm / gc;
        if ratio < 1.0 {
            if ratio < 0.01 { 0.0 } else { self.eps_e * ratio.powf(self.p_fwd - 2.0) }
        } else {
            self.eps_e
        }
    }

    /// Number of state variables for the ODE spread solver.
    /// State: [msw, mej, r, theta_cell, beta_gamma_sq, u2_th, t_comv]
    const ODE_NVAR: usize = 7;

    /// ODE right-hand side for a single cell with lateral spreading.
    /// State: [msw, mej, r, theta_cell, beta_gamma_sq, u2_th, t_comv]
    /// Evolves primitives directly — no root-finding needed.
    fn ode_rhs_spread_cell(
        &self,
        state: &[f64; Self::ODE_NVAR],
        t: f64,
        theta0_i: f64,
        theta_c: f64,
        spread: bool,
    ) -> [f64; Self::ODE_NVAR] {
        let msw = state[0];
        let mej = state[1];
        let r = state[2];
        let theta_cell = state[3];
        let bg_sq = state[4].max(0.0);
        let u2_th = state[5];

        let gamma = (bg_sq + 1.0).sqrt();
        let beta = (bg_sq / (bg_sq + 1.0)).sqrt();
        let u = bg_sq.sqrt();

        // dr/dt = β·c (fluid/contact-discontinuity velocity in lab frame)
        let dr_dt = beta * C_SPEED;

        // Mass sweeping: dm2/dt = r²·ρ·dr/dt (per steradian, NO f_spread)
        let rho = self.tool.solve_density(r) * MASS_P;
        let dm2_dt = dr_dt * rho * r * r;
        let dmsw_dt = dm2_dt;
        let dmej_dt = 0.0;

        // Spreading: dθ/dt = cs_factor · β·c / (2Γr) · f_suppress
        // cs_factor = sqrt((2u²+3)/(4u²+3)) — relativistic sound speed correction
        let dtheta_dt = if spread {
            let theta_s = theta_cell.max(theta_c);
            let f_suppress = 1.0 / (1.0 + 7.0 * u * theta_s);
            let cs_factor = ((2.0 * bg_sq + 3.0) / (4.0 * bg_sq + 3.0)).sqrt();
            f_suppress * dr_dt / (2.0 * gamma * r) * cs_factor
        } else {
            0.0
        };

        // Adiabatic index: γ_ad = 4/3 + 1/(3Γ) (Synge EOS)
        let ad_idx = reverse_shock::adiabatic_idx(gamma);

        // Effective Gamma: Γ_eff = (γ_ad·(Γ²-1) + 1) / Γ
        let gamma2 = gamma * gamma;
        let gamma_eff = (ad_idx * (gamma2 - 1.0) + 1.0) / gamma;
        let d_gamma_eff = (ad_idx * (gamma2 + 1.0) - 1.0) / gamma2;

        // Base dlnV/dt = 3·dr/dt/r (radial volume expansion)
        let mut dln_v_dt_gamma = 3.0 * dr_dt / r;

        // Spreading corrections for dGamma/dt
        let (dm_dt_swept, m_swept, u_spread) = if spread {
            let cos_theta = theta_cell.cos();
            let sin_theta = theta_cell.sin();
            let d_omega0 = (1.0 - theta0_i.cos()).max(1e-30);
            let f_spread = (1.0 - cos_theta) / d_omega0;

            // Lateral expansion factor
            let lateral = if (1.0 - cos_theta).abs() > 1e-30 {
                sin_theta / (1.0 - cos_theta) * dtheta_dt
            } else {
                0.0
            };

            // Effective mass sweeping rate (per initial solid angle)
            let dm_sw = dm2_dt * f_spread + msw / d_omega0 * sin_theta * dtheta_dt;
            let m_sw = msw * f_spread;

            // Volume expansion includes lateral term
            dln_v_dt_gamma += lateral;

            // Internal energy scaled by f_spread for dGamma equation
            let u_sp = u2_th * f_spread;

            (dm_sw, m_sw, u_sp)
        } else {
            (dm2_dt, msw, u2_th)
        };

        // dΓ/dt = (a1 + a2) / (b1 + b2)  [energy-momentum conservation]
        let mut a1 = -(gamma - 1.0) * (gamma_eff + 1.0) * C2 * dm_dt_swept;
        let a2 = (ad_idx - 1.0) * gamma_eff * u_spread * dln_v_dt_gamma;

        // Magnetar spin-down injection: L_inj [erg/s/sr]
        let l_inj: f64 = self.magnetar_episodes.iter().map(|&(l0_c2, t0, q, ts)| {
            if t >= ts {
                l0_c2 * (1.0 + (t - ts) / t0).powf(-q)
            } else {
                0.0
            }
        }).sum();
        a1 += l_inj * C2;

        let b1 = (mej + m_swept) * C2;
        let b2 = (d_gamma_eff + gamma_eff * (ad_idx - 1.0) / gamma) * u_spread;

        let d_gamma_dt = if (b1 + b2).abs() > 1e-60 {
            (a1 + a2) / (b1 + b2)
        } else {
            0.0
        };

        // d(βΓ)²/dt = 2·Γ·dΓ/dt
        let dbg_sq_dt = 2.0 * gamma * d_gamma_dt;

        // Thermal energy evolution: dU2_th/dt
        // Uses different spreading corrections than dGamma/dt (VegasAfterglow convention)
        let du2th_dt = {
            let mut dm_dt_u = dm2_dt;
            let mut dln_v_u = 3.0 * dr_dt / r - d_gamma_dt / gamma;

            if spread {
                let cos_theta = theta_cell.cos();
                let sin_theta = theta_cell.sin();

                let factor = if (1.0 - cos_theta).abs() > 1e-30 {
                    sin_theta / (1.0 - cos_theta) * dtheta_dt
                } else {
                    0.0
                };

                dm_dt_u += msw * factor;
                dln_v_u += factor;
                dln_v_u += factor / (ad_idx - 1.0);
            }

            let eps_rad = self.compute_eps_rad(gamma, state[6], rho);
            (1.0 - eps_rad) * (gamma - 1.0) * C2 * dm_dt_u - (ad_idx - 1.0) * dln_v_u * u2_th
        };

        // Comoving time: dt_comv/dt_lab = 1/Γ
        let dt_comv_dt = 1.0 / gamma;

        [dmsw_dt, dmej_dt, dr_dt, dtheta_dt, dbg_sq_dt, du2th_dt, dt_comv_dt]
    }

    /// Adaptive RK45 step for a single spreading cell.
    /// Returns (new_state, new_dt, succeeded).
    fn ode_spread_step_rk45(
        &self,
        state: &[f64; Self::ODE_NVAR],
        t: f64,
        dt: f64,
        theta0_i: f64,
        theta_c: f64,
        rtol: f64,
        spread: bool,
    ) -> ([f64; Self::ODE_NVAR], f64, bool) {
        const N: usize = SimBox::ODE_NVAR;

        let k1 = self.ode_rhs_spread_cell(state, t, theta0_i, theta_c, spread);

        let mut s2 = [0.0; N];
        for j in 0..N { s2[j] = state[j] + k1[j] * dt * 2.0 / 9.0; }
        let k2 = self.ode_rhs_spread_cell(&s2, t + dt * 2.0 / 9.0, theta0_i, theta_c, spread);

        let mut s3 = [0.0; N];
        for j in 0..N { s3[j] = state[j] + k1[j] * dt / 12.0 + k2[j] * dt / 4.0; }
        let k3 = self.ode_rhs_spread_cell(&s3, t + dt / 3.0, theta0_i, theta_c, spread);

        let mut s4 = [0.0; N];
        for j in 0..N {
            s4[j] = state[j] + k1[j] * dt * 69.0 / 128.0
                - k2[j] * dt * 243.0 / 128.0
                + k3[j] * dt * 135.0 / 64.0;
        }
        let k4 = self.ode_rhs_spread_cell(&s4, t + dt * 3.0 / 4.0, theta0_i, theta_c, spread);

        let mut s5 = [0.0; N];
        for j in 0..N {
            s5[j] = state[j] - k1[j] * dt * 17.0 / 12.0
                + k2[j] * dt * 27.0 / 4.0
                - k3[j] * dt * 27.0 / 5.0
                + k4[j] * dt * 16.0 / 15.0;
        }
        let k5 = self.ode_rhs_spread_cell(&s5, t + dt, theta0_i, theta_c, spread);

        let mut s6 = [0.0; N];
        for j in 0..N {
            s6[j] = state[j] + k1[j] * dt * 65.0 / 432.0
                - k2[j] * dt * 5.0 / 16.0
                + k3[j] * dt * 13.0 / 16.0
                + k4[j] * dt * 4.0 / 27.0
                + k5[j] * dt * 5.0 / 144.0;
        }
        let k6 = self.ode_rhs_spread_cell(&s6, t + dt * 5.0 / 6.0, theta0_i, theta_c, spread);

        // 5th-order solution
        let mut result = [0.0; N];
        for j in 0..N {
            result[j] = state[j]
                + k1[j] * dt * 47.0 / 450.0
                + k3[j] * dt * 12.0 / 25.0
                + k4[j] * dt * 32.0 / 225.0
                + k5[j] * dt / 30.0
                + k6[j] * dt * 6.0 / 25.0;
        }

        // Error estimate
        let mut rerror = 0.0f64;
        for j in 0..N {
            let error = (k1[j] * dt / 150.0
                - k3[j] * dt * 3.0 / 100.0
                + k4[j] * dt * 16.0 / 75.0
                + k5[j] * dt / 20.0
                - k6[j] * dt * 6.0 / 25.0)
                .abs();
            let scale = result[j].abs().max(1e-30);
            rerror = rerror.max(error / scale);
        }

        let succeeded = rerror < rtol;
        let boost = (0.9 * (rtol / rerror.max(1e-30)).powf(0.2)).min(1.5).max(0.2);
        let new_dt = dt * boost;

        if succeeded {
            (result, new_dt, true)
        } else {
            (*state, new_dt, false)
        }
    }

    /// Solve using per-cell ODE spreading (VegasAfterglow-style).
    /// Each theta cell evolves independently with adaptive RK45.
    /// Evolves primitive variables directly — no root-finding in the RHS.
    fn solve_ode_spread(&mut self) {
        let real_ntheta = self.ntheta;
        let n_core = self.count_uniform_core();
        let is_tophat = n_core > 1;

        // Pre-compute log-spaced output times
        let log_tmin = self.tmin.ln();
        let log_tmax = self.tmax.ln();
        let n_output = Self::ADAPTIVE_OUTPUT_POINTS;
        let mut output_times = Vec::with_capacity(n_output + 1);
        output_times.push(self.tmin);
        for k in 1..=n_output {
            let log_t = log_tmin + (log_tmax - log_tmin) * k as f64 / n_output as f64;
            output_times.push(log_t.exp());
        }
        let nt = output_times.len();

        // Initialize output: ys[7][ntheta][nt]
        let mut ys = vec![vec![vec![0.0; nt]; real_ntheta]; 7];
        let ts = output_times.clone();

        let theta_c = self.theta_c;
        let rtol = 1e-6;

        // For tophat: solve 1 representative core cell, then all tail cells.
        // For structured jet: solve all cells independently.
        let solve_list: Vec<usize> = if is_tophat {
            // Cell 0 represents the core (will be replicated to 0..n_core).
            // Then solve tail cells n_core..real_ntheta individually.
            let mut v = vec![0usize];
            for i in n_core..real_ntheta {
                v.push(i);
            }
            v
        } else {
            (0..real_ntheta).collect()
        };

        for &i in &solve_list {
            // For tophat core cells, use theta_c as the representative initial angle.
            // Using theta[0] (the first tiny grid cell) gives a pathologically
            // large f_spread = (1-cos(theta))/(1-cos(theta0)) when theta0 ≈ 0.
            let theta0_i = if is_tophat && i < n_core { self.theta_c } else { self.theta[i] };

            // Initial u2_th: thermal energy per solid angle
            let bg_sq_0 = self.beta_gamma_sq[i];
            let gamma_0 = (bg_sq_0 + 1.0).sqrt();
            let u2_th_0 = (1.0 - self.eps_e) * (gamma_0 - 1.0) * self.msw[i] * C2;
            // Initial t_comv = t_lab / Gamma (coasting phase)
            let t_comv_0 = self.tmin / gamma_0;

            // Initial state: [msw, mej, r, theta_cell, beta_gamma_sq, u2_th, t_comv]
            let mut state = [
                self.msw[i],
                self.mej[i],
                self.r[i],
                theta0_i,
                self.beta_gamma_sq[i],
                u2_th_0,
                t_comv_0,
            ];

            // Save initial primitives
            let bt = self.compute_beta_th_from_state(&state, theta0_i, true);
            ys[0][i][0] = state[0]; // msw
            ys[1][i][0] = state[1]; // mej
            ys[2][i][0] = state[4]; // beta_gamma_sq
            ys[3][i][0] = bt;       // beta_th
            ys[4][i][0] = state[2]; // r
            ys[5][i][0] = state[5]; // u2_th
            ys[6][i][0] = state[6]; // t_comv

            let mut t = self.tmin;
            let mut dt = (self.tmax - self.tmin) * 1e-4;
            let mut save_idx = 1;

            while t < self.tmax && save_idx < nt {
                let dt_max = (output_times[save_idx] - t).min(self.tmax - t + 1e-6);
                dt = dt.min(dt_max);

                let (new_state, new_dt, succeeded) =
                    self.ode_spread_step_rk45(&state, t, dt, theta0_i, theta_c, rtol, true);

                if succeeded {
                    t += dt;
                    state = new_state;
                    state[3] = state[3].max(0.0).min(PI); // clamp theta
                    state[4] = state[4].max(0.0);         // clamp bg_sq
                    state[5] = state[5].max(0.0);         // clamp u2_th

                    while save_idx < nt && output_times[save_idx] <= t + 1e-10 {
                        let bt = self.compute_beta_th_from_state(&state, theta0_i, true);
                        ys[0][i][save_idx] = state[0]; // msw
                        ys[1][i][save_idx] = state[1]; // mej
                        ys[2][i][save_idx] = state[4]; // beta_gamma_sq
                        ys[3][i][save_idx] = bt;
                        ys[4][i][save_idx] = state[2]; // r
                        ys[5][i][save_idx] = state[5]; // u2_th
                        ys[6][i][save_idx] = state[6]; // t_comv
                        save_idx += 1;
                    }

                    dt = new_dt;
                } else {
                    dt = new_dt;
                }
            }

            while save_idx < nt {
                let bt = self.compute_beta_th_from_state(&state, theta0_i, true);
                ys[0][i][save_idx] = state[0];
                ys[1][i][save_idx] = state[1];
                ys[2][i][save_idx] = state[4];
                ys[3][i][save_idx] = bt;
                ys[4][i][save_idx] = state[2];
                ys[5][i][save_idx] = state[5];
                ys[6][i][save_idx] = state[6];
                save_idx += 1;
            }
        }

        if is_tophat {
            // Replicate core cell 0 solution to all core cells (0..n_core)
            for var in 0..7 {
                let template = ys[var][0].clone();
                for j in 1..n_core {
                    ys[var][j] = template.clone();
                }
            }
        }

        self.ys = ys;
        self.ts = ts;
    }

    /// Compute beta_th from ODE state for saving.
    fn compute_beta_th_from_state(&self, state: &[f64; Self::ODE_NVAR], _theta0_i: f64, spread: bool) -> f64 {
        if !spread {
            return 0.0;
        }
        let r = state[2];
        let theta_cell = state[3];
        let bg_sq = state[4].max(0.0);

        let gamma = (bg_sq + 1.0).sqrt();
        let beta = (bg_sq / (bg_sq + 1.0)).sqrt();
        let u = bg_sq.sqrt();
        let theta_s = theta_cell.max(self.theta_c);
        let f_suppress = 1.0 / (1.0 + 7.0 * u * theta_s);
        let cs_factor = ((2.0 * bg_sq + 3.0) / (4.0 * bg_sq + 3.0)).sqrt();
        let dtheta_dt = f_suppress * beta * C_SPEED / (2.0 * gamma * r) * cs_factor;
        r * dtheta_dt / C_SPEED
    }

    /// Solve without lateral spreading using per-cell primitive-variable RK45.
    /// Reuses the ODE spread stepper with spread=false, eliminating all
    /// conservative-to-primitive root-finding and heap allocations.
    fn solve_no_spread(&mut self) {
        let real_ntheta = self.ntheta;
        let n_core = self.count_uniform_core();
        let is_tophat = n_core > 1;

        // Pre-compute log-spaced output times
        let log_tmin = self.tmin.ln();
        let log_tmax = self.tmax.ln();
        let n_output = Self::ADAPTIVE_OUTPUT_POINTS;
        let mut output_times = Vec::with_capacity(n_output + 1);
        output_times.push(self.tmin);
        for k in 1..=n_output {
            let log_t = log_tmin + (log_tmax - log_tmin) * k as f64 / n_output as f64;
            output_times.push(log_t.exp());
        }
        let nt = output_times.len();

        // Initialize output: ys[7][ntheta][nt]
        let mut ys = vec![vec![vec![0.0; nt]; real_ntheta]; 7];
        let ts = output_times.clone();

        let theta_c = self.theta_c;
        let rtol = 1e-6;

        let solve_list: Vec<usize> = if is_tophat {
            let mut v = vec![0usize];
            for i in n_core..real_ntheta {
                v.push(i);
            }
            v
        } else {
            (0..real_ntheta).collect()
        };

        for &i in &solve_list {
            let theta0_i = self.theta[i];

            // Initial u2_th: thermal energy per solid angle
            let bg_sq_0 = self.beta_gamma_sq[i];
            let gamma_0 = (bg_sq_0 + 1.0).sqrt();
            let u2_th_0 = (1.0 - self.eps_e) * (gamma_0 - 1.0) * self.msw[i] * C2;
            let t_comv_0 = self.tmin / gamma_0;

            // Initial state: [msw, mej, r, theta_cell, beta_gamma_sq, u2_th, t_comv]
            let mut state = [
                self.msw[i],
                self.mej[i],
                self.r[i],
                theta0_i,
                self.beta_gamma_sq[i],
                u2_th_0,
                t_comv_0,
            ];

            // Save initial primitives (beta_th = 0 for no-spread)
            ys[0][i][0] = state[0]; // msw
            ys[1][i][0] = state[1]; // mej
            ys[2][i][0] = state[4]; // beta_gamma_sq
            ys[3][i][0] = 0.0;      // beta_th
            ys[4][i][0] = state[2]; // r
            ys[5][i][0] = state[5]; // u2_th
            ys[6][i][0] = state[6]; // t_comv

            let mut t = self.tmin;
            let mut dt = (self.tmax - self.tmin) * 1e-4;
            let mut save_idx = 1;

            while t < self.tmax && save_idx < nt {
                let dt_max = (output_times[save_idx] - t).min(self.tmax - t + 1e-6);
                dt = dt.min(dt_max);

                let (new_state, new_dt, succeeded) =
                    self.ode_spread_step_rk45(&state, t, dt, theta0_i, theta_c, rtol, false);

                if succeeded {
                    t += dt;
                    state = new_state;
                    state[4] = state[4].max(0.0); // clamp bg_sq
                    state[5] = state[5].max(0.0); // clamp u2_th

                    while save_idx < nt && output_times[save_idx] <= t + 1e-10 {
                        ys[0][i][save_idx] = state[0]; // msw
                        ys[1][i][save_idx] = state[1]; // mej
                        ys[2][i][save_idx] = state[4]; // beta_gamma_sq
                        ys[3][i][save_idx] = 0.0;      // beta_th
                        ys[4][i][save_idx] = state[2]; // r
                        ys[5][i][save_idx] = state[5]; // u2_th
                        ys[6][i][save_idx] = state[6]; // t_comv
                        save_idx += 1;
                    }

                    dt = new_dt;
                } else {
                    dt = new_dt;
                }
            }

            while save_idx < nt {
                ys[0][i][save_idx] = state[0];
                ys[1][i][save_idx] = state[1];
                ys[2][i][save_idx] = state[4];
                ys[3][i][save_idx] = 0.0;
                ys[4][i][save_idx] = state[2];
                ys[5][i][save_idx] = state[5];
                ys[6][i][save_idx] = state[6];
                save_idx += 1;
            }
        }

        if is_tophat {
            for var in 0..7 {
                let template = ys[var][0].clone();
                for j in 1..n_core {
                    ys[var][j] = template.clone();
                }
            }
        }

        self.ys = ys;
        self.ts = ts;
    }
}
