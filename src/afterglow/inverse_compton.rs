use crate::constants::*;
use crate::afterglow::blast::{Blast, ShockType};
use crate::afterglow::models::Dict;
use crate::afterglow::ssa;

// ---------------------------------------------------------------------------
// Klein-Nishina cross-section ratio σ_KN / σ_T
// ---------------------------------------------------------------------------

/// Compton cross-section ratio σ_KN/σ_T as function of x = hν/(m_e c²).
pub fn compton_cross_section_ratio(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < 1e-2 {
        // Thomson limit
        return 1.0 - 2.0 * x;
    }
    if x > 1e2 {
        // Ultra-relativistic limit
        return 3.0 / 8.0 * (2.0 * x).ln() + 0.5 / x;
    }
    // Full Klein-Nishina formula
    let l = (1.0 + 2.0 * x).ln();
    let invx = 1.0 / x;
    let invx2 = invx * invx;
    let term1 = 1.0 + 2.0 * x;
    let invt1 = 1.0 / term1;
    let invt1_2 = invt1 * invt1;

    let a = (1.0 + x) * invx2 * invx;
    let b = 2.0 * x * (1.0 + x) * invt1 - l;
    let c = 0.5 * l * invx;
    let d = (1.0 + 3.0 * x) * invt1_2;

    0.75 * (a * b + c - d)
}

/// KN correction factor for a photon at frequency ν seen by electron with Lorentz factor γ.
pub fn compton_correction(gamma: f64, nu: f64) -> f64 {
    let x = H_PLANCK * gamma * nu / (MASS_E * C_SPEED * C_SPEED);
    compton_cross_section_ratio(x)
}

// ---------------------------------------------------------------------------
// InverseComptonY: Y(γ) parameter with KN corrections
// ---------------------------------------------------------------------------

/// Piecewise power-law segment for Y(γ).
#[derive(Clone, Debug)]
struct YSegment {
    gamma_break: f64,
    slope: f64,
}

/// Inverse Compton Y parameter evaluator.
/// Computes Y(γ) as a piecewise power-law with KN corrections.
#[derive(Clone, Debug)]
pub struct InverseComptonY {
    pub gamma_m_hat: f64,  // Thomson/KN transition for γ_m
    pub gamma_c_hat: f64,  // Thomson/KN transition for γ_c
    pub gamma_self: f64,   // SSC dominance threshold
    pub gamma0: f64,       // where Y(γ₀) = 1
    pub y_t: f64,          // Thomson Y parameter
    pub regime: usize,     // operating regime (0-5)

    // Piecewise segments (from low to high γ)
    segments: Vec<YSegment>,
    y_base: f64,           // Y value at the first segment
}

impl Default for InverseComptonY {
    fn default() -> Self {
        InverseComptonY {
            gamma_m_hat: 1.0,
            gamma_c_hat: 1.0,
            gamma_self: 1.0,
            gamma0: f64::INFINITY,
            y_t: 0.0,
            regime: 0,
            segments: Vec::new(),
            y_base: 0.0,
        }
    }
}

impl InverseComptonY {
    /// Create a new IC Y evaluator.
    pub fn new(gamma_m: f64, gamma_c: f64, p: f64, b: f64, y_t: f64, is_kn: bool) -> Self {
        let mut ic = InverseComptonY::default();
        ic.y_t = y_t;

        if b <= 0.0 || gamma_m <= 0.0 {
            return ic;
        }

        let nu_m = ssa::compute_syn_freq(gamma_m, b);
        ic.gamma_m_hat = (MASS_E * C_SPEED * C_SPEED / (H_PLANCK * nu_m)).max(1.0);
        ic.gamma_self = (ic.gamma_m_hat * gamma_m * gamma_m).powf(1.0 / 3.0);

        if !is_kn {
            // Thomson regime: Y(γ) = Y_T (constant)
            ic.regime = 0;
            ic.y_base = y_t;
            return ic;
        }

        ic.update_cooling_breaks(gamma_c, y_t, gamma_m, p);
        ic
    }

    /// Update cooling breaks and rebuild segment table.
    pub fn update_cooling_breaks(&mut self, gamma_c: f64, y_t: f64, gamma_m: f64, p: f64) {
        self.y_t = y_t;
        self.segments.clear();

        if gamma_m <= 0.0 || self.gamma_m_hat <= 0.0 {
            self.regime = 0;
            self.y_base = y_t;
            return;
        }

        self.gamma_c_hat = (self.gamma_self * self.gamma_self * self.gamma_self
            / (gamma_c * gamma_c)).max(1.0);

        if gamma_m <= gamma_c {
            // Slow cooling
            self.regime = 1;
            self.y_base = y_t;
            self.segments.push(YSegment { gamma_break: self.gamma_c_hat, slope: 0.5 * (p - 3.0) });
            self.segments.push(YSegment { gamma_break: self.gamma_m_hat, slope: -4.0 / 3.0 });
        } else if gamma_m > gamma_c && self.gamma_m_hat > gamma_m {
            // Fast cooling, weak KN
            self.regime = 2;
            self.y_base = y_t;
            self.segments.push(YSegment { gamma_break: self.gamma_m_hat, slope: -0.5 });
            self.segments.push(YSegment { gamma_break: self.gamma_c_hat, slope: -4.0 / 3.0 });
        } else {
            // Fast cooling, strong KN
            self.regime = 3;
            self.y_base = y_t;
            self.segments.push(YSegment { gamma_break: self.gamma_m_hat, slope: -1.0 / 3.0 });
            self.segments.push(YSegment { gamma_break: self.gamma_self, slope: -1.0 });
            self.segments.push(YSegment { gamma_break: self.gamma_c_hat, slope: -4.0 / 3.0 });
        }
    }

    /// Evaluate Y(γ).
    pub fn gamma_spectrum(&self, gamma: f64) -> f64 {
        if self.regime == 0 || self.segments.is_empty() {
            return self.y_t;
        }

        let mut y: f64 = self.y_base;
        let mut gamma_prev: f64 = 1.0;

        for seg in &self.segments {
            if gamma <= seg.gamma_break {
                break;
            }
            let gamma_start: f64 = gamma_prev.max(1.0);
            if gamma_start < seg.gamma_break {
                y *= (seg.gamma_break / gamma_start.max(1.0_f64)).powf(seg.slope);
            }
            gamma_prev = seg.gamma_break;
        }

        // Apply remaining scaling if gamma is above last break
        if !self.segments.is_empty() && gamma > self.segments.last().unwrap().gamma_break {
            let last = self.segments.last().unwrap();
            y *= (gamma / last.gamma_break).powf(last.slope);
        } else {
            // Find which segment gamma falls in
            let mut gamma_start: f64 = 1.0;
            for seg in &self.segments {
                if gamma < seg.gamma_break {
                    y *= (gamma / gamma_start.max(1.0_f64)).powf(seg.slope);
                    break;
                }
                y *= (seg.gamma_break / gamma_start.max(1.0_f64)).powf(seg.slope);
                gamma_start = seg.gamma_break;
            }
        }

        y.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Thomson Y parameter computation
// ---------------------------------------------------------------------------

/// Radiative efficiency in Thomson regime.
fn eta_rad_thomson(gamma_m: f64, gamma_c: f64, p: f64) -> f64 {
    if gamma_c < gamma_m {
        // Fast cooling
        1.0
    } else {
        // Slow cooling
        if gamma_m <= 0.0 { return 0.0; }
        (gamma_c / gamma_m).powf(2.0 - p).min(1.0)
    }
}

/// Compute Thomson Y parameter: Y = 0.5 * (√(1 + 4b) - 1) where b = η_e ε_e / ε_B.
pub fn compute_thomson_y(eps_e: f64, eps_b: f64, gamma_m: f64, gamma_c: f64, p: f64) -> f64 {
    let eta_e = eta_rad_thomson(gamma_m, gamma_c, p);
    let b = eta_e * eps_e / eps_b;
    0.5 * ((1.0 + 4.0 * b).sqrt() - 1.0)
}

// ---------------------------------------------------------------------------
// Iterative γ_c update with Thomson Y
// ---------------------------------------------------------------------------

/// Update γ_c iteratively with Thomson Y parameter.
pub fn update_gamma_c_thomson(
    t_comv: f64,
    b: f64,
    gamma_m: f64,
    eps_e: f64,
    eps_b: f64,
    p: f64,
) -> (f64, f64, InverseComptonY) {
    let mut gamma_c = ssa::compute_gamma_c(t_comv, b, 0.0);
    let mut y_t = compute_thomson_y(eps_e, eps_b, gamma_m, gamma_c, p);

    for _ in 0..100 {
        let gamma_c_new = ssa::compute_gamma_c(t_comv, b, y_t);
        if (gamma_c_new - gamma_c).abs() / gamma_c.max(1.0) < 1e-3 {
            gamma_c = gamma_c_new;
            break;
        }
        gamma_c = gamma_c_new;
        y_t = compute_thomson_y(eps_e, eps_b, gamma_m, gamma_c, p);
    }

    let ys = InverseComptonY::new(gamma_m, gamma_c, p, b, y_t, false);
    (gamma_c, y_t, ys)
}

/// Update γ_c iteratively with Klein-Nishina corrections.
pub fn update_gamma_c_kn(
    t_comv: f64,
    b: f64,
    gamma_m: f64,
    eps_e: f64,
    eps_b: f64,
    p: f64,
) -> (f64, f64, InverseComptonY) {
    let mut gamma_c = ssa::compute_gamma_c(t_comv, b, 0.0);
    let mut y_t = compute_thomson_y(eps_e, eps_b, gamma_m, gamma_c, p);
    let mut ys = InverseComptonY::new(gamma_m, gamma_c, p, b, y_t, true);

    for _ in 0..100 {
        let y_c = ys.gamma_spectrum(gamma_c);
        let gamma_c_new = ssa::compute_gamma_c(t_comv, b, y_c);
        if (gamma_c_new - gamma_c).abs() / gamma_c.max(1.0) < 1e-3 {
            gamma_c = gamma_c_new;
            break;
        }
        gamma_c = gamma_c_new;
        y_t = compute_thomson_y(eps_e, eps_b, gamma_m, gamma_c, p);
        ys.update_cooling_breaks(gamma_c, y_t, gamma_m, p);
    }

    (gamma_c, y_t, ys)
}

// ---------------------------------------------------------------------------
// SSC synchrotron model
// ---------------------------------------------------------------------------

/// Synchrotron self-Compton model: synchrotron + IC with optional KN corrections.
/// Parameters: eps_e, eps_b, p, plus optional "ssc_kn" (0 or 1).
pub fn sync_ssc(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_ssc requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_ssc requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_ssc requires 'p'");
    let use_kn = p.get("ssc_kn").copied().unwrap_or(0.0) > 0.5;

    // Get B-field and density
    let (b, n_blast, t_comv, gamma_th, dr) = match blast.shock_type {
        ShockType::Forward => {
            let e = blast.e_density;
            let b = (8.0 * PI * eps_b * e).sqrt();
            (b, blast.n_blast, blast.t_comv, blast.gamma_th, blast.dr)
        }
        ShockType::Reverse => {
            (blast.b3, blast.n3, blast.t_comv, blast.gamma_th3, blast.dr)
        }
    };

    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 || t_comv <= 0.0 || gamma_th <= 1.0 {
        return 0.0;
    }

    // Compute gamma_m
    let gamma_m_max = ssa::compute_syn_gamma_m_max(b, 0.0);
    let gamma_m = ssa::compute_syn_gamma_m(gamma_th, gamma_m_max, eps_e, p_val, 1.0);

    // Iterative gamma_c with IC cooling
    let (gamma_c, y_t, _ys) = if use_kn {
        update_gamma_c_kn(t_comv, b, gamma_m, eps_e, eps_b, p_val)
    } else {
        update_gamma_c_thomson(t_comv, b, gamma_m, eps_e, eps_b, p_val)
    };

    // Compute synchrotron spectrum with IC-corrected γ_c
    let nu_m = ssa::compute_syn_freq(gamma_m, b);
    let nu_c = ssa::compute_syn_freq(gamma_c, b);
    let e_p = PITCH_ANGLE_AVG * 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b * n_blast
        / MASS_E / C_SPEED / C_SPEED;

    let syn_emissivity = if nu_m < nu_c {
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

    // IC contribution: approximate as Y_T × synchrotron at ν/γ_m²
    // This is a simplified SSC model; full computation would require
    // convolution over the electron distribution
    let nu_ic = nu / (gamma_m * gamma_m * 4.0 / 3.0);
    let ic_emissivity = if y_t > 0.0 && nu_ic > 0.0 {
        let syn_seed = if nu_ic < nu_m {
            e_p * (nu_ic / nu_m).cbrt()
        } else if nu_ic < nu_c {
            e_p * (nu_ic / nu_m).powf(-(p_val - 1.0) / 2.0)
        } else {
            e_p * (nu_c / nu_m).powf(-(p_val - 1.0) / 2.0) * (nu_ic / nu_c).powf(-p_val / 2.0)
        };
        y_t * syn_seed
    } else {
        0.0
    };

    (syn_emissivity + ic_emissivity) * dr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compton_cross_section_thomson_limit() {
        // At x << 1: σ_KN/σ_T ≈ 1
        let ratio = compton_cross_section_ratio(1e-5);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compton_cross_section_kn_regime() {
        // At x >> 1: σ_KN/σ_T << 1
        let ratio = compton_cross_section_ratio(100.0);
        assert!(ratio < 0.1);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_compton_cross_section_monotonic() {
        // σ_KN/σ_T should decrease with increasing x
        let r1 = compton_cross_section_ratio(0.1);
        let r2 = compton_cross_section_ratio(1.0);
        let r3 = compton_cross_section_ratio(10.0);
        assert!(r1 > r2);
        assert!(r2 > r3);
    }

    #[test]
    fn test_thomson_y() {
        let y = compute_thomson_y(0.1, 0.01, 100.0, 1000.0, 2.3);
        assert!(y > 0.0);
        assert!(y.is_finite());
    }

    #[test]
    fn test_ic_y_thomson_regime() {
        let ic = InverseComptonY::new(100.0, 1000.0, 2.3, 0.1, 1.0, false);
        assert_eq!(ic.regime, 0);
        // In Thomson regime, Y should be constant
        let y1 = ic.gamma_spectrum(10.0);
        let y2 = ic.gamma_spectrum(100.0);
        assert!((y1 - y2).abs() < 1e-10);
    }

    #[test]
    fn test_update_gamma_c_converges() {
        let (gc, yt, _) = update_gamma_c_thomson(1e5, 0.1, 100.0, 0.1, 0.01, 2.3);
        assert!(gc > 1.0);
        assert!(gc.is_finite());
        assert!(yt > 0.0);
    }

    #[test]
    fn test_sync_ssc_positive() {
        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.3);

        let blast = Blast {
            t: 1e5,
            gamma: 10.0,
            gamma_th: 10.0,
            n_blast: 1e3,
            e_density: 1e-2,
            dr: 1e15,
            t_comv: 1e4,
            ..Blast::default()
        };

        let flux = sync_ssc(1e14, &p, &blast);
        assert!(flux >= 0.0);
        assert!(flux.is_finite());
    }
}
