// Global constants in CGS units
pub const C_SPEED: f64 = 29979245800.0;
pub const MASS_P: f64 = 1.672622e-24;
pub const MASS_E: f64 = 9.109384e-28;
pub const SIGMA_T: f64 = 6.6524587e-25;
pub const E_CHARGE: f64 = 4.803204673e-10;
pub const MPC: f64 = 3.09e24;
pub const PI: f64 = std::f64::consts::PI;
pub const H_PLANCK: f64 = 6.626070e-27; // erg·s (Planck constant)
pub const MAS: f64 = 1.0 / 206264806.24709466;

/// Synchrotron pitch angle averaging factor: ⟨sin α⟩ × F(x)_max.
/// ⟨sin α⟩ = π/4 for isotropic pitch angle distribution,
/// F(x)_max = 0.92 is the peak of the single-electron synchrotron spectrum
/// (Rybicki & Lightman 1979, §6.2).
pub const PITCH_ANGLE_AVG: f64 = std::f64::consts::FRAC_PI_4 * 0.92;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_of_light() {
        assert!((C_SPEED - 2.998e10).abs() / C_SPEED < 1e-3);
    }

    #[test]
    fn test_proton_mass() {
        assert!((MASS_P - 1.673e-24).abs() / MASS_P < 1e-3);
    }

    #[test]
    fn test_electron_mass() {
        assert!((MASS_E - 9.109e-28).abs() / MASS_E < 1e-3);
    }

    #[test]
    fn test_pi() {
        assert!((PI - std::f64::consts::PI).abs() < 1e-15);
    }
}
