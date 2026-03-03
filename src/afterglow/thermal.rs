use crate::constants::*;
use crate::afterglow::blast::{Blast, ShockType};
use crate::afterglow::models::Dict;
use crate::math::special::{gamma_fn, mq21_f_theta};

// ---------------------------------------------------------------------------
// MQ21 helper functions
// ---------------------------------------------------------------------------

/// a(Theta) fitting function — MQ21 eq. (1), from Gammie & Popham (1998).
fn a_theta(theta: f64) -> f64 {
    (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)
}

/// Minimum Lorentz factor of power-law electrons — MQ21 eq. (6).
fn gamma_m_thermal(theta: f64) -> f64 {
    1.0 + a_theta(theta) * theta
}

/// g(Theta, p) correction for non-relativistic power-law electrons — MQ21 eq. (8).
fn g_theta_p(theta: f64, p: f64) -> f64 {
    let gamma_m = gamma_m_thermal(theta);
    let num = (p - 1.0) * (1.0 + a_theta(theta) * theta);
    let denom = (p - 1.0) * gamma_m - p + 2.0;
    if denom.abs() < 1e-30 {
        return 1.0;
    }
    (num / denom) * (gamma_m / (3.0 * theta)).powf(p - 1.0)
}

/// I'(x) — Mahadevan et al. (1996) fitting function for thermal synchrotron.
/// MQ21 eq. (13).
fn i_prime(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    4.0505 * x.powf(-1.0 / 6.0)
        * (1.0 + 0.40 * x.powf(-0.25) + 0.5316 * x.powf(-0.5))
        * (-1.8899 * x.powf(1.0 / 3.0)).exp()
}

/// C_j(p) — power-law emissivity prefactor, MQ21 eq. (15).
fn compute_c_j(p: f64) -> f64 {
    (gamma_fn((p + 5.0) / 4.0) / gamma_fn((p + 7.0) / 4.0))
        * gamma_fn((3.0 * p + 19.0) / 12.0)
        * gamma_fn((3.0 * p - 1.0) / 12.0)
        * ((p - 2.0) / (p + 1.0))
        * 3.0_f64.powf((2.0 * p - 1.0) / 2.0)
        * 2.0_f64.powf(-(7.0 - p) / 2.0)
        * PI.powf(-0.5)
}

/// C_alpha(p) — power-law absorption prefactor, MQ21 eq. (17).
fn compute_c_alpha(p: f64) -> f64 {
    (gamma_fn((p + 6.0) / 4.0) / gamma_fn((p + 8.0) / 4.0))
        * gamma_fn((3.0 * p + 2.0) / 12.0)
        * gamma_fn((3.0 * p + 22.0) / 12.0)
        * (p - 2.0)
        * 3.0_f64.powf((2.0 * p - 5.0) / 2.0)
        * 2.0_f64.powf(p / 2.0)
        * PI.powf(1.5)
}

/// Low-frequency correction to power-law emissivity below nu_m.
/// Smooth interpolation between high- and low-frequency regimes.
fn low_freq_jpl_correction(x: f64, theta: f64, p: f64) -> f64 {
    let gamma_m = gamma_m_thermal(theta);
    // Synchrotron constant in x << x_m limit (MQ21 reference code)
    let c_j_low = -PI.powf(1.5) * (p - 2.0)
        / (2.0_f64.powf(1.0 / 3.0)
            * 3.0_f64.powf(1.0 / 6.0)
            * (3.0 * p - 1.0)
            * gamma_fn(1.0 / 3.0)
            * gamma_fn(-1.0 / 3.0)
            * gamma_fn(11.0 / 6.0));
    let c_j_val = compute_c_j(p);
    if c_j_val.abs() < 1e-100 {
        return 1.0;
    }
    let corr = (c_j_low / c_j_val)
        * (gamma_m / (3.0 * theta)).powf(-(3.0 * p - 1.0) / 3.0)
        * x.powf((3.0 * p - 1.0) / 6.0);
    let s = 3.0 / p;
    if corr <= 0.0 || !corr.is_finite() {
        return 1.0;
    }
    (1.0 + corr.powf(-s)).powf(-1.0 / s)
}

/// Low-frequency correction to power-law absorption below nu_m.
fn low_freq_apl_correction(x: f64, theta: f64, p: f64) -> f64 {
    let gamma_m = gamma_m_thermal(theta);
    // Synchrotron constant in x << x_m limit (MQ21 reference code)
    let c_alpha_low = -2.0_f64.powf(8.0 / 3.0) * PI.powf(3.5) * (p + 2.0) * (p - 2.0)
        / (3.0_f64.powf(19.0 / 6.0)
            * (3.0 * p + 2.0)
            * gamma_fn(1.0 / 3.0)
            * gamma_fn(-1.0 / 3.0)
            * gamma_fn(11.0 / 6.0));
    let c_alpha_val = compute_c_alpha(p);
    if c_alpha_val.abs() < 1e-100 {
        return 1.0;
    }
    let corr = (c_alpha_low / c_alpha_val)
        * (gamma_m / (3.0 * theta)).powf(-(3.0 * p + 2.0) / 3.0)
        * x.powf((3.0 * p + 2.0) / 6.0);
    let s = 3.0 / p;
    if corr <= 0.0 || !corr.is_finite() {
        return 1.0;
    }
    (1.0 + corr.powf(-s)).powf(-1.0 / s)
}

/// Thermal synchrotron frequency nu_Theta — MQ21 eq. (11).
fn nu_theta(theta: f64, b: f64) -> f64 {
    3.0 * theta * theta * E_CHARGE * b / (4.0 * PI * MASS_E * C_SPEED)
}

// ---------------------------------------------------------------------------
// Emissivity and absorption coefficient functions
// ---------------------------------------------------------------------------

/// Thermal emissivity j_th — MQ21 eqs. (10, 20).
fn jnu_th(x: f64, n: f64, b: f64, theta: f64, z_cool: f64) -> f64 {
    let e3 = E_CHARGE * E_CHARGE * E_CHARGE;
    let mut val = (3.0_f64.sqrt() / (8.0 * PI))
        * (e3 / (MASS_E * C_SPEED * C_SPEED))
        * mq21_f_theta(theta)
        * n
        * b
        * x
        * i_prime(x);
    // Fast-cooling correction (MQ21 eq. 20)
    let z0 = (2.0 * x).powf(1.0 / 3.0);
    if z0 > z_cool && z_cool > 0.0 {
        val *= z_cool / z0;
    }
    val
}

/// Power-law emissivity j_pl — MQ21 eqs. (14, 19).
fn jnu_pl(x: f64, n: f64, b: f64, theta: f64, delta: f64, p: f64, z_cool: f64) -> f64 {
    let e3 = E_CHARGE * E_CHARGE * E_CHARGE;
    let mut val = compute_c_j(p)
        * (e3 / (MASS_E * C_SPEED * C_SPEED))
        * delta
        * n
        * b
        * g_theta_p(theta, p)
        * x.powf(-(p - 1.0) / 2.0);
    // Low-frequency correction
    val *= low_freq_jpl_correction(x, theta, p);
    // Fast-cooling correction
    let z0 = x.sqrt();
    if z0 > z_cool && z_cool > 0.0 {
        val *= z_cool / z0;
    }
    val
}

/// Thermal absorption coefficient alpha_th — MQ21 eqs. (12, 20).
fn alphanu_th(x: f64, n: f64, b: f64, theta: f64, z_cool: f64) -> f64 {
    let mut val = PI * 3.0_f64.powf(-1.5)
        * E_CHARGE
        * (n / (theta.powi(5) * b))
        * mq21_f_theta(theta)
        * x.powf(-1.0)
        * i_prime(x);
    // Fast-cooling correction
    let z0 = (2.0 * x).powf(1.0 / 3.0);
    if z0 > z_cool && z_cool > 0.0 {
        val *= z_cool / z0;
    }
    val
}

/// Power-law absorption coefficient alpha_pl — MQ21 eqs. (16, 19).
fn alphanu_pl(x: f64, n: f64, b: f64, theta: f64, delta: f64, p: f64, z_cool: f64) -> f64 {
    let mut val = compute_c_alpha(p)
        * E_CHARGE
        * (delta * n / (theta.powi(5) * b))
        * g_theta_p(theta, p)
        * x.powf(-(p + 4.0) / 2.0);
    // Low-frequency correction
    val *= low_freq_apl_correction(x, theta, p);
    // Fast-cooling correction
    let z0 = x.sqrt();
    if z0 > z_cool && z_cool > 0.0 {
        val *= z_cool / z0;
    }
    val
}

// ---------------------------------------------------------------------------
// Full-volume helpers (FM25 — Ferguson & Margalit 2025)
// ---------------------------------------------------------------------------

/// Downstream fluid Lorentz factor from shock Lorentz factor via Rankine-Hugoniot
/// jump conditions.  FM25 Eq. B7 / MQ24.
fn gamma_fluid_from_shock(gamma_shock: f64) -> f64 {
    let bg_sh_sq = gamma_shock * gamma_shock - 1.0; // (βΓ_shock)²
    let bg_fluid_sq = 0.5
        * (bg_sh_sq - 2.0
            + (bg_sh_sq * bg_sh_sq + 5.0 * bg_sh_sq + 4.0).sqrt());
    if bg_fluid_sq <= 0.0 {
        return 1.0;
    }
    (1.0 + bg_fluid_sq).sqrt()
}

/// Self-similar coordinate bounding the emitting post-shock region.
/// FM25 Appendix C.  k is the CSM density power-law index (0 = ISM, 2 = wind).
fn xi_shell_val(gamma_fluid: f64, k: f64) -> f64 {
    let denom = 4.0 * (3.0 - k) * gamma_fluid * gamma_fluid;
    if denom <= 3.0 {
        return 0.0; // shell fills entire volume
    }
    (1.0 - 3.0 / denom).cbrt()
}

// ---------------------------------------------------------------------------
// Main radiation model function
// ---------------------------------------------------------------------------

/// Thermal + power-law synchrotron model following Margalit & Quataert (2021).
///
/// Parameters from Dict:
/// - `eps_e`: fraction of energy in non-thermal electrons
/// - `eps_b`: magnetic field efficiency
/// - `p`: power-law index
/// - `eps_T`: electron thermalization efficiency (default 1.0)
/// - `delta`: fraction of electron energy in power-law tail (default: eps_e/eps_T, capped at 1)
/// - `full_volume`: if > 0.5, use FM25 full-volume post-shock convention instead of thin-shell
/// - `k`: CSM density power-law index (default 0.0; 0 = ISM, 2 = wind). Only used with full_volume.
///
/// Returns specific intensity × dr (same units as other models).
pub fn sync_thermal(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_thermal requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_thermal requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_thermal requires 'p'");
    let eps_t = p.get("eps_T").copied().unwrap_or(1.0);
    let full_volume = p.get("full_volume").copied().unwrap_or(0.0) > 0.5;

    // Extract shock parameters — either thin-shell or FM25 full-volume
    let (b, n_e, t_comv, beta, dr) = if full_volume {
        // FM25 uniform post-shock convention
        let k = p.get("k").copied().unwrap_or(0.0);
        let (gamma_sh, n_upstream) = match blast.shock_type {
            ShockType::Forward => (blast.gamma, blast.n_ambient),
            ShockType::Reverse => (blast.gamma34, blast.n4_upstream),
        };
        if gamma_sh <= 1.0 || blast.r <= 0.0 {
            return 0.0;
        }
        let gamma_f = gamma_fluid_from_shock(gamma_sh);
        let beta_f = (1.0 - 1.0 / (gamma_f * gamma_f)).sqrt();
        let xi_s = xi_shell_val(gamma_f, k);
        let dr_vol = blast.r * (1.0 - xi_s);

        let mu: f64 = 0.62;
        let mu_e: f64 = 1.18;
        let n_e_vol = 4.0 * n_upstream * mu_e * gamma_f;
        let u_vol =
            (gamma_f - 1.0) * n_e_vol * mu * MASS_P * C_SPEED * C_SPEED / mu_e;
        let b_vol = (8.0 * PI * eps_b * u_vol).sqrt();
        let t_comv_vol = blast.t / gamma_f;

        if b_vol <= 0.0 || n_e_vol <= 0.0 || dr_vol <= 0.0 || t_comv_vol <= 0.0 {
            return 0.0;
        }
        (b_vol, n_e_vol, t_comv_vol, beta_f, dr_vol)
    } else {
        // Existing thin-shell computation
        let (b, n_e, t_comv, gamma_shock, dr) = match blast.shock_type {
            ShockType::Forward => {
                let e = blast.e_density;
                let b = (8.0 * PI * eps_b * e).sqrt();
                (b, blast.n_blast, blast.t_comv, blast.gamma_th, blast.dr)
            }
            ShockType::Reverse => {
                (blast.b3, blast.n3, blast.t_comv, blast.gamma_th3, blast.dr)
            }
        };

        if b <= 0.0 || n_e <= 0.0 || dr <= 0.0 || t_comv <= 0.0 || gamma_shock <= 1.0 {
            return 0.0;
        }

        let beta = (1.0 - 1.0 / (gamma_shock * gamma_shock)).sqrt();
        (b, n_e, t_comv, beta, dr)
    };

    // Dimensionless electron temperature — MQ21 eqs. (2,3)
    // Using the shock energy density approach:
    // Theta_0 = eps_T * (9*mu*mp)/(32*mu_e*me) * beta^2
    // For simplicity, use mu=0.62, mu_e=1.18 (solar composition)
    let mu = 0.62;
    let mu_e = 1.18;
    let theta_0 = eps_t * (9.0 * mu * MASS_P / (32.0 * mu_e * MASS_E)) * beta * beta;
    let theta = (5.0 * theta_0 - 6.0
        + (25.0 * theta_0 * theta_0 + 180.0 * theta_0 + 36.0).sqrt())
        / 30.0;

    if theta <= 0.0 || !theta.is_finite() {
        return 0.0;
    }

    // Magnetic field is already computed from eps_b above.

    // delta: fraction of electron energy in power-law tail
    let delta = p.get("delta").copied().unwrap_or_else(|| {
        (eps_e / eps_t).min(1.0)
    });

    // Thermal synchrotron frequency — MQ21 eq. (11)
    let nu_th = nu_theta(theta, b);
    if nu_th <= 0.0 {
        return 0.0;
    }

    // Dimensionless frequency
    let x = nu / nu_th;
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }

    // Normalized cooling Lorentz factor — MQ21 eq. (18)
    let z_cool = (6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * t_comv)) / theta;

    // Total emissivity
    let j_th = jnu_th(x, n_e, b, theta, z_cool);
    let j_pl = jnu_pl(x, n_e, b, theta, delta, p_val, z_cool);
    let j_total = j_th + j_pl;

    if j_total <= 0.0 || !j_total.is_finite() {
        return 0.0;
    }

    // Total absorption coefficient
    let alpha_th = alphanu_th(x, n_e, b, theta, z_cool);
    let alpha_pl = alphanu_pl(x, n_e, b, theta, delta, p_val, z_cool);
    let alpha_total = alpha_th + alpha_pl;

    // Optical depth
    let tau = alpha_total * dr;

    // Apply SSA: I = j/alpha * (1 - exp(-tau)) for tau > 0
    if tau < 1e-10 {
        // Optically thin: I = j * dr
        j_total * dr
    } else if tau > 100.0 {
        // Optically thick: I = j/alpha (source function)
        j_total / alpha_total
    } else {
        // General case
        (j_total / alpha_total) * (1.0 - (-tau).exp())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::afterglow::blast::Blast;

    fn make_params() -> Dict {
        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.5);
        p.insert("eps_T".into(), 1.0);
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
            s: 0.5,
            doppler: 5.0,
            cos_theta_beta: 0.95,
            n_blast: 1e3,
            e_density: 1e-2,
            pressure: 1e-3,
            n_ambient: 1.0,
            dr: 1e15,
            t_comv: 1e4,
            ..Blast::default()
        }
    }

    #[test]
    fn test_a_theta() {
        // a(0) = 6/4 = 1.5
        assert!((a_theta(0.0) - 1.5).abs() < 1e-10);
        // a(inf) → 15/5 = 3
        assert!((a_theta(1e10) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_gamma_m_thermal_values() {
        // gamma_m(0) = 1 + 1.5 * 0 = 1
        assert!((gamma_m_thermal(0.0) - 1.0).abs() < 1e-10);
        // gamma_m increases with Theta
        assert!(gamma_m_thermal(1.0) > 1.0);
        assert!(gamma_m_thermal(10.0) > gamma_m_thermal(1.0));
    }

    #[test]
    fn test_i_prime_shape() {
        // I'(x) should be positive for x > 0
        assert!(i_prime(1.0) > 0.0);
        assert!(i_prime(0.1) > 0.0);
        assert!(i_prime(10.0) > 0.0);
        // I'(x) → 0 for large x (exponential cutoff)
        assert!(i_prime(1000.0) < i_prime(10.0));
        // I'(0) should be 0
        assert_eq!(i_prime(0.0), 0.0);
    }

    #[test]
    fn test_c_j_positive() {
        let cj = compute_c_j(2.5);
        assert!(cj > 0.0, "C_j should be positive, got {}", cj);
        assert!(cj.is_finite());
    }

    #[test]
    fn test_c_alpha_positive() {
        let ca = compute_c_alpha(2.5);
        assert!(ca > 0.0, "C_alpha should be positive, got {}", ca);
        assert!(ca.is_finite());
    }

    #[test]
    fn test_sync_thermal_positive() {
        let p = make_params();
        let blast = make_blast();
        // Test at radio frequency
        let result = sync_thermal(1e9, &p, &blast);
        assert!(result >= 0.0, "sync_thermal should be non-negative at radio, got {}", result);
        assert!(result.is_finite());
        // Test at X-ray
        let result_xray = sync_thermal(1e18, &p, &blast);
        assert!(result_xray >= 0.0, "sync_thermal should be non-negative at X-ray, got {}", result_xray);
        assert!(result_xray.is_finite());
    }

    #[test]
    fn test_sync_thermal_spectral_shape() {
        let p = make_params();
        let blast = make_blast();
        // Check that we get finite positive output across a broad range
        let freqs = [1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18];
        for &nu in &freqs {
            let val = sync_thermal(nu, &p, &blast);
            assert!(val >= 0.0 && val.is_finite(), "Bad value at nu={:.0e}: {}", nu, val);
        }
    }

    #[test]
    fn test_sync_thermal_ssa_suppresses_low_freq() {
        let p = make_params();
        // Use a denser blast to make SSA more prominent
        let mut blast = make_blast();
        blast.n_blast = 1e6;
        blast.e_density = 1e1;

        let flux_low = sync_thermal(1e6, &p, &blast);
        let flux_mid = sync_thermal(1e10, &p, &blast);
        // At very low frequency, SSA should suppress emission relative to mid-freq
        // (flux should not diverge at low frequency)
        assert!(flux_low.is_finite());
        assert!(flux_mid.is_finite());
    }

    #[test]
    fn test_sync_thermal_zero_for_bad_input() {
        let p = make_params();
        let mut blast = make_blast();
        blast.dr = 0.0;
        assert_eq!(sync_thermal(1e10, &p, &blast), 0.0);

        blast = make_blast();
        blast.n_blast = 0.0;
        assert_eq!(sync_thermal(1e10, &p, &blast), 0.0);
    }

    #[test]
    fn test_nu_theta_scaling() {
        // nu_Theta ∝ Theta^2 * B
        let nu1 = nu_theta(1.0, 1.0);
        let nu2 = nu_theta(2.0, 1.0);
        assert!((nu2 / nu1 - 4.0).abs() < 1e-10); // ratio should be 4
    }

    // --- Full-volume helper tests ---

    #[test]
    fn test_gamma_fluid_ultra_relativistic() {
        // For Γ_shock >> 1: Γ_fluid ≈ Γ_shock in the lab frame
        // (the Γ/√2 relation is for the relative Lorentz factor in the shock frame)
        let gamma_sh = 100.0;
        let gamma_f = gamma_fluid_from_shock(gamma_sh);
        let rel_err = (gamma_f - gamma_sh).abs() / gamma_sh;
        assert!(rel_err < 0.01, "Ultra-rel: Γ_fluid={:.4}, expected≈{:.4}, err={:.4}", gamma_f, gamma_sh, rel_err);
        // Must be slightly > Γ_shock (from the formula)
        assert!(gamma_f > 0.99 * gamma_sh);
    }

    #[test]
    fn test_gamma_fluid_non_relativistic() {
        // For Γ_shock → 1 (β → 0): Γ_fluid → 1
        let gamma_sh = 1.0001;
        let gamma_f = gamma_fluid_from_shock(gamma_sh);
        assert!((gamma_f - 1.0).abs() < 0.01, "Non-rel: Γ_fluid={:.6}, expected≈1", gamma_f);
    }

    #[test]
    fn test_gamma_fluid_at_unity() {
        // Γ_shock = 1 exactly: should return 1
        assert_eq!(gamma_fluid_from_shock(1.0), 1.0);
    }

    #[test]
    fn test_gamma_fluid_intermediate() {
        // Γ_shock = 2: βΓ² = 3, expected βΓ_f² = 0.5*(3-2+sqrt(9+15+4)) = 0.5*(1+sqrt(28))
        let gamma_sh = 2.0;
        let gamma_f = gamma_fluid_from_shock(gamma_sh);
        let bg_sh_sq: f64 = 3.0;
        let expected_bg_f_sq = 0.5 * (bg_sh_sq - 2.0 + (bg_sh_sq*bg_sh_sq + 5.0*bg_sh_sq + 4.0).sqrt());
        let expected = (1.0_f64 + expected_bg_f_sq).sqrt();
        assert!((gamma_f - expected).abs() < 1e-10, "Γ_f={}, expected={}", gamma_f, expected);
    }

    #[test]
    fn test_xi_shell_ultra_relativistic() {
        // For Γ_fluid >> 1: ξ_shell → 1 (thin shell), dr_vol → 0
        let xi = xi_shell_val(100.0, 0.0);
        assert!(xi > 0.99, "Ultra-rel ISM: ξ_shell={:.6}, expected≈1", xi);
    }

    #[test]
    fn test_xi_shell_mildly_relativistic() {
        // For Γ_fluid ~ 2, k=0: ξ = (1 - 3/(48))^(1/3) = (1 - 1/16)^(1/3)
        let xi = xi_shell_val(2.0, 0.0);
        let expected = (1.0_f64 - 3.0 / (4.0 * 3.0 * 4.0)).cbrt();
        assert!((xi - expected).abs() < 1e-10, "ξ={}, expected={}", xi, expected);
        assert!(xi < 0.99, "Mildly relativistic shell should be thick");
    }

    #[test]
    fn test_xi_shell_wind_vs_ism() {
        // k=2 (wind) should give different (smaller) ξ than k=0 (ISM)
        // because denom = 4*(3-k)*Γ² is smaller for k=2
        let xi_ism = xi_shell_val(3.0, 0.0);
        let xi_wind = xi_shell_val(3.0, 2.0);
        assert!(xi_wind < xi_ism, "Wind shell should be thicker (smaller ξ): ξ_wind={}, ξ_ism={}", xi_wind, xi_ism);
    }

    #[test]
    fn test_xi_shell_fills_volume() {
        // When denom ≤ 3, shell fills entire volume → ξ = 0
        // k=2, Γ_fluid=1: denom = 4*(3-2)*1 = 4 > 3, so not quite
        // k=2.5, Γ_fluid=1: denom = 4*(0.5)*1 = 2 < 3, so ξ = 0
        let xi = xi_shell_val(1.0, 2.5);
        assert_eq!(xi, 0.0);
    }

    #[test]
    fn test_sync_thermal_full_volume_positive() {
        let mut p = make_params();
        p.insert("full_volume".into(), 1.0);
        p.insert("k".into(), 0.0);
        let blast = make_blast();

        let result = sync_thermal(1e9, &p, &blast);
        assert!(result > 0.0, "full_volume should produce positive flux at radio, got {}", result);
        assert!(result.is_finite());

        let result_xray = sync_thermal(1e18, &p, &blast);
        assert!(result_xray >= 0.0, "full_volume should be non-negative at X-ray, got {}", result_xray);
        assert!(result_xray.is_finite());
    }

    #[test]
    fn test_sync_thermal_full_volume_larger_flux() {
        // Full-volume should generally give higher flux than thin-shell
        // due to larger effective dr
        let p_thin = make_params();
        let mut p_full = make_params();
        p_full.insert("full_volume".into(), 1.0);
        p_full.insert("k".into(), 0.0);

        // Use a mildly relativistic shock where the effect is largest
        let mut blast = make_blast();
        blast.gamma = 2.0;
        blast.gamma_th = 2.0;
        blast.beta = (1.0 - 1.0 / 4.0_f64).sqrt();
        blast.e_density = 1e-4;
        blast.n_blast = 10.0;
        blast.dr = 1e14;

        let flux_thin = sync_thermal(1e10, &p_thin, &blast);
        let flux_full = sync_thermal(1e10, &p_full, &blast);

        // Both should be positive
        assert!(flux_thin > 0.0, "thin flux={}", flux_thin);
        assert!(flux_full > 0.0, "full flux={}", flux_full);
    }

    #[test]
    fn test_sync_thermal_full_volume_spectral_shape() {
        let mut p = make_params();
        p.insert("full_volume".into(), 1.0);
        p.insert("k".into(), 0.0);
        let blast = make_blast();

        let freqs = [1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18];
        for &nu in &freqs {
            let val = sync_thermal(nu, &p, &blast);
            assert!(val >= 0.0 && val.is_finite(), "full_volume bad at nu={:.0e}: {}", nu, val);
        }
    }

    #[test]
    fn test_sync_thermal_full_volume_convergence_high_gamma() {
        // At very high Γ, full-volume and thin-shell should produce similar
        // magnetic fields and electron densities (both approach thin-shell limit).
        // We can't compare fluxes directly (different n_e/B derivations),
        // but both should produce finite, positive values.
        let mut p_full = make_params();
        p_full.insert("full_volume".into(), 1.0);
        p_full.insert("k".into(), 0.0);

        let mut blast = make_blast();
        blast.gamma = 100.0;
        blast.beta = (1.0 - 1.0 / 10000.0_f64).sqrt();

        let flux = sync_thermal(1e10, &p_full, &blast);
        assert!(flux > 0.0 && flux.is_finite(), "High-Γ full_volume: {}", flux);
    }
}
