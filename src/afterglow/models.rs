use std::collections::HashMap;
use crate::constants::*;
use crate::afterglow::blast::Blast;

pub type Dict = HashMap<String, f64>;
pub type RadiationModel = fn(nu: f64, p: &Dict, blast: &Blast) -> f64;

/// Standard synchrotron model (Sari 1998).
pub fn sync(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync requires 'eps_b'");
    let p_val = *p.get("p").expect("sync requires 'p'");

    let n_blast = blast.n_blast;
    let t = blast.t;
    let gamma = blast.gamma;
    let e = blast.e_density;

    let gamma_m = (p_val - 2.0) / (p_val - 1.0) * (eps_e * MASS_P / MASS_E * (gamma - 1.0));
    let b = (8.0 * PI * eps_b * e).sqrt();
    let gamma_c = 6.0 * PI * MASS_E * gamma * C_SPEED / SIGMA_T / b / b / t;
    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * n_blast
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

    emissivity * blast.dr
}

/// Synchrotron with deep Newtonian phase correction.
pub fn sync_dnp(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_dnp requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_dnp requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_dnp requires 'p'");

    let n_blast = blast.n_blast;
    let t = blast.t;
    let gamma = blast.gamma;
    let e = blast.e_density;

    let mut gamma_m = (p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (gamma - 1.0);
    let mut f = 1.0;
    if gamma_m <= 1.0 {
        f = (p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (gamma - 1.0);
        gamma_m = 1.0;
    }
    let b = (8.0 * PI * eps_b * e).sqrt();
    let gamma_c = 6.0 * PI * MASS_E * gamma * C_SPEED / SIGMA_T / b / b / t;
    let nu_m = 3.0 * E_CHARGE * b * gamma_m * gamma_m / 4.0 / PI / C_SPEED / MASS_E;
    let nu_c = 3.0 * E_CHARGE * b * gamma_c * gamma_c / 4.0 / PI / C_SPEED / MASS_E;
    let e_p = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * f * n_blast
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

    emissivity * blast.dr
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
        "sync_dnp" => Some(sync_dnp),
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
}
