/// Which shock region this blast element represents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShockType {
    Forward,
    Reverse,
}

impl Default for ShockType {
    fn default() -> Self {
        ShockType::Forward
    }
}

/// Blast fluid element properties.
#[derive(Clone, Default)]
pub struct Blast {
    // coordinate values (burster frame)
    pub t: f64,
    pub theta: f64,
    pub phi: f64,
    pub r: f64,

    // blast velocity (burster frame)
    pub beta: f64,
    pub gamma: f64,
    pub beta_th: f64,
    pub beta_r: f64,
    pub beta_f: f64,
    pub gamma_f: f64,
    pub s: f64,

    // angles (burster frame)
    pub doppler: f64,
    pub cos_theta_beta: f64,

    // thermodynamic values (comoving frame)
    pub n_blast: f64,
    pub e_density: f64,
    pub gamma_th: f64,
    pub pressure: f64,
    pub n_ambient: f64,
    pub dr: f64,

    // shock type
    pub shock_type: ShockType,

    // reverse shock specific fields
    pub gamma_th3: f64,    // thermal Lorentz factor in region 3
    pub b3: f64,           // magnetic field in region 3
    pub n3: f64,           // number density in region 3
    pub t_comv: f64,       // comoving time
    pub gamma34: f64,      // relative Lorentz factor between regions 4 and 3
    pub n4_upstream: f64,  // upstream ejecta number density (region 4)
    pub gamma_c_override: f64,  // post-crossing cooled gamma_c (0 = not set)
    pub gamma_M_override: f64,  // post-crossing cooled gamma_M (0 = not set)
}
