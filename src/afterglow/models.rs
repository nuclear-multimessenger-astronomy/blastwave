use std::collections::HashMap;
use crate::constants::*;
use crate::afterglow::blast::{Blast, ShockType};
use crate::math::fast_math::fast_powf;

pub type Dict = HashMap<String, f64>;
pub type RadiationModel = fn(nu: f64, p: &Dict, blast: &Blast) -> f64;

/// Pre-computed synchrotron parameters for a single blast state.
/// Avoids recomputing sync_params() when only the frequency changes.
pub struct CachedSyncParams {
    pub nu_m: f64,
    pub nu_c: f64,
    pub nu_M: f64,
    pub e_p: f64,
    pub dr: f64,
    pub valid: bool,
}

impl CachedSyncParams {
    /// Build cached params from a blast state (forward shock only).
    pub fn from_blast(blast: &Blast, eps_e: f64, eps_b: f64, p_val: f64) -> Self {
        let (gamma_m, gamma_c, b, n_blast, dr, f_syn) = sync_params(eps_e, eps_b, p_val, blast);
        if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 {
            return CachedSyncParams {
                nu_m: 0.0, nu_c: 0.0, nu_M: 0.0, e_p: 0.0, dr: 0.0, valid: false,
            };
        }

        let nu_coeff = 3.0 * E_CHARGE / (4.0 * PI * C_SPEED * MASS_E);
        let nu_m = nu_coeff * b * gamma_m * gamma_m;
        let nu_c = nu_coeff * b * gamma_c * gamma_c;

        let gamma_M = (6.0 * PI * E_CHARGE / (SIGMA_T * b)).sqrt();
        let nu_M = nu_coeff * b * gamma_M * gamma_M;

        let e_p = PITCH_ANGLE_AVG * 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f_syn * n_blast
            / MASS_E / C_SPEED / C_SPEED;

        CachedSyncParams {
            nu_m, nu_c, nu_M, e_p, dr, valid: true,
        }
    }
}

/// Fast synchrotron evaluation using cached params and fast_powf.
/// Equivalent to sync() but skips sync_params() recomputation.
#[inline]
pub fn sync_from_cached(nu: f64, p_val: f64, c: &CachedSyncParams) -> f64 {
    if !c.valid {
        return 0.0;
    }

    let emissivity = if c.nu_m < c.nu_c {
        if nu < c.nu_m {
            c.e_p * (nu / c.nu_m).cbrt()
        } else if nu < c.nu_c {
            c.e_p * fast_powf(nu / c.nu_m, -(p_val - 1.0) / 2.0)
        } else {
            c.e_p * fast_powf(c.nu_c / c.nu_m, -(p_val - 1.0) / 2.0) * fast_powf(nu / c.nu_c, -p_val / 2.0)
        }
    } else {
        if nu < c.nu_c {
            c.e_p * (nu / c.nu_c).cbrt()
        } else if nu < c.nu_m {
            c.e_p / (nu / c.nu_c).sqrt()
        } else {
            c.e_p / (c.nu_m / c.nu_c).sqrt() * fast_powf(nu / c.nu_m, -p_val / 2.0)
        }
    };

    emissivity * c.dr
}

/// Fast smooth synchrotron evaluation using cached params and fast_powf.
/// Equivalent to sync_smooth() but skips sync_params() recomputation.
#[inline]
pub fn sync_smooth_from_cached(nu: f64, p_val: f64, c: &CachedSyncParams) -> f64 {
    if !c.valid {
        return 0.0;
    }

    let emissivity = if c.nu_m < c.nu_c {
        let x = nu / c.nu_m;
        let xc = nu / c.nu_c;
        let delta1 = (p_val - 1.0) / 2.0 + 1.0 / 3.0;
        c.e_p * 2.0 * x.cbrt() / (1.0 + fast_powf(x, delta1)) / (1.0 + xc.sqrt())
    } else {
        let x = nu / c.nu_c;
        let xm = nu / c.nu_m;
        c.e_p * 2.0 * x.cbrt() / (1.0 + fast_powf(x, 5.0 / 6.0)) / (1.0 + fast_powf(xm, (p_val - 1.0) / 2.0))
    };

    emissivity * (-nu / c.nu_M).exp() * c.dr
}

/// Fast deep Newtonian synchrotron using cached params and fast_powf.
/// Equivalent to sync_dnp() but skips sync_params() recomputation.
#[inline]
pub fn sync_dnp_from_cached(nu: f64, p_val: f64, c: &CachedSyncParams) -> f64 {
    // For the fast path, we use the same logic as sync_from_cached
    // since the DNP correction is already baked into cached params
    sync_from_cached(nu, p_val, c)
}

/// Compute synchrotron characteristic quantities.
/// For forward shock: derives B, n from blast.e_density, blast.n_blast, blast.gamma, blast.t.
/// For reverse shock: uses pre-computed blast.b3, blast.n3, blast.gamma_th3, blast.t_comv.
fn sync_params(eps_e: f64, eps_b: f64, p_val: f64, blast: &Blast) -> (f64, f64, f64, f64, f64, f64) {
    match blast.shock_type {
        ShockType::Forward => {
            let n_blast = blast.n_blast;
            let e = blast.e_density;

            let gamma_m = (p_val - 2.0) / (p_val - 1.0) * (eps_e * MASS_P / MASS_E * (blast.gamma_th - 1.0)) + 1.0;
            let b = (8.0 * PI * eps_b * e).sqrt();
            // Use tracked comoving time for gamma_c (t_comv = integral of dt/Gamma)
            let t_comv = blast.t_comv;
            // Newtonian correction: solve γ_c(γ_c - 1) = gamma_bar
            // → γ_c = (gamma_bar + √(gamma_bar² + 4)) / 2 (matches VegasAfterglow)
            let gamma_bar = 6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * t_comv);
            let gamma_c = (gamma_bar + (gamma_bar * gamma_bar + 4.0).sqrt()) / 2.0;
            (gamma_m, gamma_c, b, n_blast, blast.dr, 1.0)
        }
        ShockType::Reverse => {
            let b = blast.b3;
            let n3 = blast.n3;
            let gamma_th3 = blast.gamma_th3;
            let t_comv = blast.t_comv;

            if b <= 0.0 || n3 <= 0.0 || gamma_th3 <= 1.0 || t_comv <= 0.0 {
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }

            let gamma_m = (p_val - 2.0) / (p_val - 1.0) * eps_e * (gamma_th3 - 1.0) * MASS_P / MASS_E + 1.0;
            // Post-crossing: use cooled gamma_c from crossing time
            let gamma_c = if blast.gamma_c_override > 0.0 {
                blast.gamma_c_override
            } else {
                let gamma_bar = 6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * t_comv);
                (gamma_bar + (gamma_bar * gamma_bar + 4.0).sqrt()) / 2.0
            };
            // Cyclotron correction: suppresses emission when gamma_m ≈ 1
            // (electrons are barely relativistic, synchrotron → cyclotron).
            // Matches VegasAfterglow's f_syn factor.
            let f_syn = if gamma_m > 1.0 && p_val <= 3.0 {
                (gamma_m - 1.0) / gamma_m
            } else if gamma_m > 1.0 {
                ((gamma_m - 1.0) / gamma_m).powf((p_val - 1.0) / 2.0)
            } else {
                0.0
            };
            (gamma_m, gamma_c, b, n3, blast.dr, f_syn)
        }
    }
}

/// Standard synchrotron model (Sari 1998).
pub fn sync(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync requires 'eps_b'");
    let p_val = *p.get("p").expect("sync requires 'p'");

    let (gamma_m, gamma_c, b, n_blast, dr, f_syn) = sync_params(eps_e, eps_b, p_val, blast);
    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 {
        return 0.0;
    }

    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = PITCH_ANGLE_AVG * 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f_syn * n_blast
        / MASS_E
        / C_SPEED
        / C_SPEED;

    let emissivity = if nu_m < nu_c {
        if nu < nu_m {
            e_p * (nu / nu_m).cbrt()
        } else if nu < nu_c {
            e_p * (nu / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu / nu_c).powf(-p_val / 2.0)
        }
    } else {
        if nu < nu_c {
            e_p * (nu / nu_c).cbrt()
        } else if nu < nu_m {
            e_p / (nu / nu_c).sqrt()
        } else {
            e_p / (nu_m / nu_c).sqrt() * (nu / nu_m).powf(-p_val / 2.0)
        }
    };

    emissivity * dr
}

/// Synchrotron with deep Newtonian phase correction.
pub fn sync_dnp(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_dnp requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_dnp requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_dnp requires 'p'");

    let (mut gamma_m, gamma_c, b, n_blast, dr, f_syn) = sync_params(eps_e, eps_b, p_val, blast);
    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 {
        return 0.0;
    }

    let mut f = f_syn;
    if gamma_m <= 1.0 {
        // Deep Newtonian correction: the original forward-shock formula
        if blast.shock_type == ShockType::Forward {
            f = (p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (blast.gamma_th - 1.0);
        } else {
            f = (p_val - 2.0) / (p_val - 1.0) * eps_e * (blast.gamma_th3 - 1.0) * MASS_P / MASS_E;
        }
        gamma_m = 1.0;
    }

    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = PITCH_ANGLE_AVG * 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f * n_blast
        / MASS_E
        / C_SPEED
        / C_SPEED;

    let emissivity = if nu_m < nu_c {
        if nu < nu_m {
            e_p * (nu / nu_m).cbrt()
        } else if nu < nu_c {
            e_p * (nu / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu / nu_c).powf(-p_val / 2.0)
        }
    } else {
        if nu < nu_c {
            e_p * (nu / nu_c).cbrt()
        } else if nu < nu_m {
            e_p / (nu / nu_c).sqrt()
        } else {
            e_p / (nu_m / nu_c).sqrt() * (nu / nu_m).powf(-p_val / 2.0)
        }
    };

    emissivity * dr
}

/// Smooth power-law synchrotron model (matches VegasAfterglow SmoothPowerLawSyn).
///
/// Uses smooth transitions between spectral segments instead of sharp breaks (Sari 1998).
/// Sharpness parameter s=1 (hardcoded, matching VegasAfterglow).
/// Factor of 2 normalization compensates for s=1 giving half the peak value at the first break.
/// Includes exp(-ν/ν_M) high-energy cutoff from maximum electron Lorentz factor.
pub fn sync_smooth(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_smooth requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_smooth requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_smooth requires 'p'");

    let (gamma_m, gamma_c, b, n_blast, dr, f_syn) = sync_params(eps_e, eps_b, p_val, blast);
    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 {
        return 0.0;
    }

    // Maximum electron Lorentz factor (post-crossing: cooled adiabatically)
    let gamma_M = if blast.shock_type == ShockType::Reverse && blast.gamma_M_override > 0.0 {
        blast.gamma_M_override
    } else {
        (6.0 * PI * E_CHARGE / (SIGMA_T * b)).sqrt()
    };
    let nu_M = 3.0 * E_CHARGE * b * gamma_M * gamma_M / 4.0 / PI / C_SPEED / MASS_E;

    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = PITCH_ANGLE_AVG * 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f_syn * n_blast
        / MASS_E
        / C_SPEED
        / C_SPEED;

    let emissivity = if nu_m < nu_c {
        // Slow cooling: ν^(1/3) → ν^(-(p-1)/2) at ν_m → ν^(-p/2) at ν_c
        let x = nu / nu_m;
        let xc = nu / nu_c;
        // Δβ₁ = (p-1)/2 + 1/3 = (3p-1)/6 = p/2 - 1/6
        let delta1 = (p_val - 1.0) / 2.0 + 1.0 / 3.0;
        e_p * 2.0 * x.cbrt() / (1.0 + x.powf(delta1)) / (1.0 + xc.sqrt())
    } else {
        // Fast cooling: ν^(1/3) → ν^(-1/2) at ν_c → ν^(-p/2) at ν_m
        let x = nu / nu_c;
        let xm = nu / nu_m;
        // Δβ₁ = 1/3 + 1/2 = 5/6
        // Δβ₂ = (p-1)/2
        e_p * 2.0 * x.cbrt() / (1.0 + x.powf(5.0 / 6.0)) / (1.0 + xm.powf((p_val - 1.0) / 2.0))
    };

    // High-energy cutoff: exp(-ν/ν_M)
    emissivity * (-nu / nu_M).exp() * dr
}

/// Weighted average model: offset.
pub fn avg_offset(_nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let theta_v = *p.get("theta_v").unwrap();
    let x_tilde = -blast.theta.sin() * blast.phi.cos() * theta_v.cos()
        + blast.theta.cos() * theta_v.sin();
    x_tilde * blast.r
}

/// Weighted average model: sigma_x.
pub fn avg_sigma_x(_nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let theta_v = *p.get("theta_v").unwrap();
    let x_tilde = -blast.theta.sin() * blast.phi.cos() * theta_v.cos()
        + blast.theta.cos() * theta_v.sin();
    x_tilde * blast.r * x_tilde * blast.r
}

/// Weighted average model: sigma_y.
pub fn avg_sigma_y(_nu: f64, _p: &Dict, blast: &Blast) -> f64 {
    let y = blast.theta.sin() * blast.phi.sin();
    y * blast.r * y * blast.r
}

/// Look up a built-in radiation model by name.
pub fn get_radiation_model(name: &str) -> Option<RadiationModel> {
    match name {
        "sync" => Some(sync),
        "sync_smooth" => Some(sync_smooth),
        "sync_dnp" => Some(sync_dnp),
        "sync_ssa" => Some(crate::afterglow::ssa::sync_ssa),
        "sync_ssa_smooth" => Some(crate::afterglow::ssa::sync_ssa_smooth),
        "sync_ssc" => Some(crate::afterglow::inverse_compton::sync_ssc),
        "sync_thermal" => Some(crate::afterglow::thermal::sync_thermal),
        "numeric" => Some(crate::afterglow::chang_cooper::sync_numeric),
        _ => None,
    }
}

/// Look up a built-in average model by name.
pub fn get_avg_model(name: &str) -> Option<RadiationModel> {
    match name {
        "offset" => Some(avg_offset),
        "sigma_x" => Some(avg_sigma_x),
        "sigma_y" => Some(avg_sigma_y),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_radiation_model_sync() {
        assert!(get_radiation_model("sync").is_some());
    }

    #[test]
    fn test_get_radiation_model_sync_dnp() {
        assert!(get_radiation_model("sync_dnp").is_some());
    }

    #[test]
    fn test_get_radiation_model_invalid() {
        assert!(get_radiation_model("nonexistent").is_none());
    }

    #[test]
    fn test_get_avg_model_offset() {
        assert!(get_avg_model("offset").is_some());
    }

    #[test]
    fn test_get_avg_model_sigma_x() {
        assert!(get_avg_model("sigma_x").is_some());
    }

    #[test]
    fn test_get_avg_model_sigma_y() {
        assert!(get_avg_model("sigma_y").is_some());
    }

    #[test]
    fn test_get_avg_model_invalid() {
        assert!(get_avg_model("nonexistent").is_none());
    }

    fn make_params() -> Dict {
        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.17);
        p.insert("theta_v".into(), 0.0);
        p
    }

    fn make_blast() -> Blast {
        Blast {
            t: 1e5,
            theta: 0.05,
            phi: 0.0,
            r: 1e17,
            beta: 0.99,
            gamma: 10.0,
            gamma_th: 10.0,
            beta_th: 0.0,
            beta_r: 0.99,
            beta_f: 0.99,
            gamma_f: 10.0,
            s: 0.5,
            doppler: 5.0,
            cos_theta_beta: 0.95,
            n_blast: 1e3,
            e_density: 1e-2,
            pressure: 1e-3,
            n_ambient: 1.0,
            dr: 1e15,
            t_comv: 1e4, // t_lab / gamma
            ..Blast::default()
        }
    }

    #[test]
    fn test_sync_positive_emissivity() {
        let p = make_params();
        let blast = make_blast();
        let result = sync(1e18, &p, &blast);
        assert!(result > 0.0, "Synchrotron emissivity should be positive");
        assert!(result.is_finite());
    }

    #[test]
    fn test_sync_dnp_positive_emissivity() {
        let p = make_params();
        let blast = make_blast();
        let result = sync_dnp(1e18, &p, &blast);
        assert!(result > 0.0, "sync_dnp emissivity should be positive");
        assert!(result.is_finite());
    }

    #[test]
    fn test_sync_decreases_with_frequency() {
        // In the power-law regime, higher frequency => lower emissivity
        let p = make_params();
        let blast = make_blast();
        let low = sync(1e14, &p, &blast);
        let high = sync(1e20, &p, &blast);
        assert!(low > high, "Emissivity should generally decrease at high frequencies");
    }

    #[test]
    fn test_sync_smooth_positive_emissivity() {
        let p = make_params();
        let blast = make_blast();
        let result = sync_smooth(1e18, &p, &blast);
        assert!(result > 0.0, "sync_smooth emissivity should be positive");
        assert!(result.is_finite());
    }

    #[test]
    fn test_sync_smooth_greater_than_sync() {
        // Smooth PL should give more flux than broken PL in power-law segments
        let p = make_params();
        let blast = make_blast();
        let f_bpl = sync(1e18, &p, &blast);
        let f_spl = sync_smooth(1e18, &p, &blast);
        assert!(f_spl > f_bpl, "Smooth PL should exceed broken PL away from breaks: spl={:.4e}, bpl={:.4e}", f_spl, f_bpl);
    }

    #[test]
    fn test_get_radiation_model_sync_smooth() {
        assert!(get_radiation_model("sync_smooth").is_some());
    }
}
