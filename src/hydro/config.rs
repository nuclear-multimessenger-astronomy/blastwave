/// Spreading mode for the hydro solver.
#[derive(Clone, Copy, PartialEq)]
pub enum SpreadMode {
    /// No lateral spreading (cells evolve independently, no θ evolution).
    None,
    /// Per-cell ODE spreading (VegasAfterglow-style, adaptive RK45).
    Ode,
    /// Finite-volume PDE spreading (CFL-limited RK2).
    Pde,
}

/// Simulation configuration, matching C++ JetConfig.
#[derive(Clone)]
pub struct JetConfig {
    pub theta_edge: Vec<f64>,
    pub eb: Vec<f64>,
    pub ht: Vec<f64>,
    pub msw: Vec<f64>,
    pub mej: Vec<f64>,
    pub r: Vec<f64>,
    pub nwind: f64,
    pub nism: f64,
    pub k: f64,
    pub tmin: f64,
    pub tmax: f64,
    pub rtol: f64,
    pub cfl: f64,
    pub spread: bool,
    pub spread_mode: SpreadMode,
    pub theta_c: f64,
    pub cal_level: i32,

    // Forward shock microphysics (affects thermal energy budget)
    pub eps_e: f64,               // electron energy fraction (forward shock); 0 = no correction
    pub eps_b: f64,               // magnetic energy fraction (forward shock); 0 = no eps_rad correction
    pub p_fwd: f64,               // electron spectral index (forward shock); used for eps_rad

    // Reverse shock parameters
    pub include_reverse_shock: bool,
    pub sigma: f64,           // magnetization parameter (0 = unmagnetized)
    pub eps_e_rs: f64,        // electron energy fraction (reverse shock)
    pub eps_b_rs: f64,        // magnetic field energy fraction (reverse shock)
    pub p_rs: f64,            // electron spectral index (reverse shock)

    // Engine duration (controls shell thickness for reverse shock)
    pub duration: f64,        // central engine duration [s] (0 = thin shell, uses tmin)

    // Energy injection parameters (reverse shock)
    pub t0_injection: f64,    // characteristic injection time (0 = no injection)
    pub l_injection: f64,     // injection luminosity (erg/s/sr)
    pub m_dot_injection: f64, // mass injection rate (g/s/sr)

    // Forward shock energy injection (magnetar spin-down)
    // Each Vec element defines one injection episode.
    // L_i(t) = l0[i] * (1 + (t - ts[i])/t0[i])^(-q[i])  for t >= ts[i], else 0
    pub magnetar_l0: Vec<f64>,    // isotropic-equivalent luminosity [erg/s] (empty = disabled)
    pub magnetar_t0: Vec<f64>,    // spin-down timescale [s]
    pub magnetar_q: Vec<f64>,     // power-law decay index (2 = magnetic dipole)
    pub magnetar_ts: Vec<f64>,    // injection start time [s] (0 = from beginning)
}

impl Default for JetConfig {
    fn default() -> Self {
        JetConfig {
            theta_edge: Vec::new(),
            eb: Vec::new(),
            ht: Vec::new(),
            msw: Vec::new(),
            mej: Vec::new(),
            r: Vec::new(),
            nwind: 0.0,
            nism: 0.0,
            k: 2.0,
            tmin: 10.0,
            tmax: 1e10,
            rtol: 1e-6,
            cfl: 0.9,
            spread: true,
            spread_mode: SpreadMode::Pde,
            theta_c: 0.1,
            cal_level: 1,
            eps_e: 0.0,
            eps_b: 0.0,
            p_fwd: 2.3,
            include_reverse_shock: false,
            sigma: 0.0,
            eps_e_rs: 0.1,
            eps_b_rs: 0.01,
            p_rs: 2.3,
            duration: 0.0,
            t0_injection: 0.0,
            l_injection: 0.0,
            m_dot_injection: 0.0,
            magnetar_l0: Vec::new(),
            magnetar_t0: Vec::new(),
            magnetar_q: Vec::new(),
            magnetar_ts: Vec::new(),
        }
    }
}
