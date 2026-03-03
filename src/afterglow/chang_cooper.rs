/// Chang-Cooper implicit finite-difference solver for the electron kinetic equation.
///
/// Evolves N(γ) on a log-spaced grid, computes synchrotron emissivity and SSA
/// from the full electron distribution. Uses the Dermer (2009) synchrotron kernel.
///
/// Reference: Chang & Cooper (1970), PyBlastAfterglowMag (Nedora et al.)

use std::cell::RefCell;
use crate::constants::*;
use crate::afterglow::blast::{Blast, ShockType};
use crate::afterglow::models::Dict;

// ---------------------------------------------------------------------------
// Synchrotron kernel — Dermer (2009) F(x) approximation
// ---------------------------------------------------------------------------

/// Synchrotron spectral function F(x) where x = ν/ν_c.
/// Dermer (2009) fitting formula, accurate to ~1% over the full range.
fn synchrotron_f(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    if x > 30.0 {
        // Exponential cutoff dominates
        return 0.0;
    }
    let x13 = x.cbrt();
    let x23 = x13 * x13;
    let x43 = x23 * x23;
    (1.808 * x13 / (1.0 + 3.4 * x23).sqrt())
        * ((1.0 + 2.21 * x23 + 0.347 * x43) / (1.0 + 1.353 * x23 + 0.217 * x43))
        * (-x).exp()
}

// ---------------------------------------------------------------------------
// Chang-Cooper Solver
// ---------------------------------------------------------------------------

pub struct ChangCooperSolver {
    n_bins: usize,
    gamma_max: f64,
    // Grid arrays
    gamma: Vec<f64>,       // log-spaced γ grid [n_bins]
    gamma_half: Vec<f64>,  // half-grid γ_{j+1/2} [n_bins-1]
    d_gamma: Vec<f64>,     // Δγ spacing [n_bins+1]
    d_gamma_bar: Vec<f64>, // averaged spacing [n_bins]
    // State
    n_e: Vec<f64>,         // electron distribution N(γ) [n_bins]
    // Work arrays
    heating: Vec<f64>,     // γ̇ at half-grid [n_bins-1]
    source: Vec<f64>,      // injection Q(γ) [n_bins]
    delta_j: Vec<f64>,     // Chang-Cooper parameter [n_bins-1]
    // Tridiagonal coefficients
    a: Vec<f64>,           // sub-diagonal [n_bins]
    b: Vec<f64>,           // diagonal [n_bins]
    c_coeff: Vec<f64>,     // super-diagonal [n_bins]
    rhs: Vec<f64>,         // right-hand side [n_bins]
    // Thomas algorithm scratch
    c_prime: Vec<f64>,
    d_prime: Vec<f64>,
}

impl ChangCooperSolver {
    pub fn new(n_bins: usize, gamma_max: f64) -> Self {
        ChangCooperSolver {
            n_bins,
            gamma_max,
            gamma: vec![0.0; n_bins],
            gamma_half: vec![0.0; n_bins.saturating_sub(1)],
            d_gamma: vec![0.0; n_bins + 1],
            d_gamma_bar: vec![0.0; n_bins],
            n_e: vec![0.0; n_bins],
            heating: vec![0.0; n_bins.saturating_sub(1)],
            source: vec![0.0; n_bins],
            delta_j: vec![0.0; n_bins.saturating_sub(1)],
            a: vec![0.0; n_bins],
            b: vec![0.0; n_bins],
            c_coeff: vec![0.0; n_bins],
            rhs: vec![0.0; n_bins],
            c_prime: vec![0.0; n_bins],
            d_prime: vec![0.0; n_bins],
        }
    }

    /// Resize if needed and rebuild the γ grid.
    pub fn resize(&mut self, n_bins: usize, gamma_max: f64) {
        if self.n_bins == n_bins && self.gamma_max == gamma_max {
            return;
        }
        self.n_bins = n_bins;
        self.gamma_max = gamma_max;
        let nb = n_bins;
        self.gamma.resize(nb, 0.0);
        self.gamma_half.resize(nb.saturating_sub(1), 0.0);
        self.d_gamma.resize(nb + 1, 0.0);
        self.d_gamma_bar.resize(nb, 0.0);
        self.n_e.resize(nb, 0.0);
        self.heating.resize(nb.saturating_sub(1), 0.0);
        self.source.resize(nb, 0.0);
        self.delta_j.resize(nb.saturating_sub(1), 0.0);
        self.a.resize(nb, 0.0);
        self.b.resize(nb, 0.0);
        self.c_coeff.resize(nb, 0.0);
        self.rhs.resize(nb, 0.0);
        self.c_prime.resize(nb, 0.0);
        self.d_prime.resize(nb, 0.0);
    }

    /// Build the log-spaced gamma grid.
    fn build_grid(&mut self) {
        let n = self.n_bins;
        let step = self.gamma_max.powf(1.0 / n as f64);
        // γ[i] = step^i (starts at 1.0)
        for i in 0..n {
            self.gamma[i] = step.powi(i as i32);
        }
        // half-grid points
        for i in 0..n - 1 {
            self.gamma_half[i] = (self.gamma[i] * self.gamma[i + 1]).sqrt();
        }
        // Δγ spacing (including boundary extrapolation)
        for i in 1..n {
            self.d_gamma[i] = self.gamma[i] - self.gamma[i - 1];
        }
        // boundary extrapolation
        self.d_gamma[0] = self.d_gamma[1];
        self.d_gamma[n] = self.d_gamma[n - 1];
        // averaged spacing
        for i in 0..n {
            self.d_gamma_bar[i] = 0.5 * (self.d_gamma[i] + self.d_gamma[i + 1]);
        }
    }

    /// Set the initial electron distribution N(γ) ∝ γ^{-p} for γ_m ≤ γ ≤ γ_max,
    /// normalized so ∫N dγ = n_total. Also sets source=0 (cooling-only evolution).
    fn inject_electrons(&mut self, p_val: f64, gamma_m: f64, n_total: f64) {
        let n = self.n_bins;
        // Compute raw distribution
        let mut integral = 0.0;
        for i in 0..n {
            if self.gamma[i] >= gamma_m && self.gamma[i] <= self.gamma_max {
                self.n_e[i] = self.gamma[i].powf(-p_val);
            } else {
                self.n_e[i] = 0.0;
            }
            integral += self.n_e[i] * self.d_gamma_bar[i];
            self.source[i] = 0.0;
        }
        // Normalize so ∫N dγ = n_total
        if integral > 0.0 {
            let norm = n_total / integral;
            for i in 0..n {
                self.n_e[i] *= norm;
            }
        }
    }

    /// Set source term for pair injection (continuous injection rate).
    fn set_source(&mut self, p_val: f64, gamma_m: f64, n_total: f64) {
        let n = self.n_bins;
        let mut integral = 0.0;
        for i in 0..n {
            if self.gamma[i] >= gamma_m && self.gamma[i] <= self.gamma_max {
                self.source[i] = self.gamma[i].powf(-p_val);
            } else {
                self.source[i] = 0.0;
            }
            integral += self.source[i] * self.d_gamma_bar[i];
        }
        if integral > 0.0 {
            let norm = n_total / integral;
            for i in 0..n {
                self.source[i] *= norm;
            }
        }
    }

    /// Compute cooling rates at half-grid points.
    /// γ̇ < 0 for cooling (energy loss).
    fn compute_cooling(&mut self, b: f64, dln_v_dt: f64) {
        let coeff_syn = SIGMA_T * b * b / (6.0 * PI * MASS_E * C_SPEED);
        let n = self.n_bins;
        for j in 0..n - 1 {
            let g = self.gamma_half[j];
            // Synchrotron cooling: dγ/dt = -(σ_T B² γ²) / (6π m_e c)
            let syn = -coeff_syn * g * g;
            // Adiabatic cooling: dγ/dt = -(dlnV/dt)(γ²-1)/(3γ)
            let adi = -dln_v_dt * (g * g - 1.0) / (3.0 * g);
            self.heating[j] = syn + adi;
        }
    }

    /// Compute Chang-Cooper parameter δ_j.
    /// δ_j = 1/w - 1/(e^w - 1) where w = Δγ · heating / dispersion.
    /// For pure advection (no dispersion), δ = 0 (upwind) if heating < 0, δ = 1 (downwind) if > 0.
    fn compute_delta(&mut self) {
        let n = self.n_bins;
        for j in 0..n - 1 {
            let h = self.heating[j];
            if h.abs() < 1e-100 {
                self.delta_j[j] = 0.5;
                continue;
            }
            // Pure advection (no diffusion): w → ±∞
            // δ = 1/w - 1/(e^w-1): upwind differencing
            let w = self.d_gamma[j + 1] * h;
            // Use numerical dispersion as regularizer; for pure cooling, w<0 → δ≈0 (upwind)
            if w.abs() > 500.0 {
                self.delta_j[j] = if w > 0.0 { 1.0 } else { 0.0 };
            } else {
                let ew = w.exp();
                if (ew - 1.0).abs() < 1e-15 {
                    self.delta_j[j] = 0.5;
                } else {
                    self.delta_j[j] = 1.0 / w - 1.0 / (ew - 1.0);
                }
            }
            self.delta_j[j] = self.delta_j[j].clamp(0.0, 1.0);
        }
    }

    /// Assemble and solve the tridiagonal system via implicit backward Euler.
    fn solve_tridiagonal(&mut self, dt: f64) {
        let n = self.n_bins;

        // Build tridiagonal coefficients
        // The Chang-Cooper scheme gives:
        // N_j^{n+1} - dt * [C_{j+1/2}((1-δ_{j})N_{j+1} + δ_j N_j) - C_{j-1/2}((1-δ_{j-1})N_j + δ_{j-1} N_{j-1})] / Δγ̄_j
        //   = N_j^n + dt * Q_j
        // where C_{j+1/2} = heating_{j} and the tridiagonal is:
        // a_j N_{j-1} + b_j N_j + c_j N_{j+1} = rhs_j

        // Zero out
        for i in 0..n {
            self.a[i] = 0.0;
            self.b[i] = 1.0;
            self.c_coeff[i] = 0.0;
            self.rhs[i] = self.n_e[i] + dt * self.source[i];
        }

        // Interior points
        for j in 1..n - 1 {
            let dg_bar = self.d_gamma_bar[j];
            if dg_bar <= 0.0 { continue; }

            // Flux at j+1/2
            let h_plus = self.heating[j]; // heating at half-grid j+1/2
            let d_plus = self.delta_j[j];

            // Flux at j-1/2
            let h_minus = self.heating[j - 1];
            let d_minus = self.delta_j[j - 1];

            // sub-diagonal: coefficient of N_{j-1}
            self.a[j] = -dt * h_minus * d_minus / dg_bar;

            // super-diagonal: coefficient of N_{j+1}
            self.c_coeff[j] = dt * h_plus * (1.0 - d_plus) / dg_bar;

            // diagonal: coefficient of N_j
            self.b[j] = 1.0 - dt * (h_plus * d_plus - h_minus * (1.0 - d_minus)) / dg_bar;
        }

        // Boundary conditions: N=0 at boundaries (already handled by b=1, a=c=0, rhs=0 at edges)
        self.rhs[0] = 0.0;
        self.rhs[n - 1] = 0.0;

        // Thomas algorithm (tridiagonal solve)
        self.c_prime[0] = self.c_coeff[0] / self.b[0];
        self.d_prime[0] = self.rhs[0] / self.b[0];

        for i in 1..n {
            // Guard against division by zero
            let denom = self.b[i] - self.a[i] * self.c_prime[i - 1];
            if denom.abs() < 1e-300 {
                self.c_prime[i] = 0.0;
                self.d_prime[i] = 0.0;
                continue;
            }
            self.c_prime[i] = self.c_coeff[i] / denom;
            self.d_prime[i] = (self.rhs[i] - self.a[i] * self.d_prime[i - 1]) / denom;
        }

        // Back substitution
        self.n_e[n - 1] = self.d_prime[n - 1];
        for i in (0..n - 1).rev() {
            self.n_e[i] = self.d_prime[i] - self.c_prime[i] * self.n_e[i + 1];
        }

        // Clamp negatives to zero
        for i in 0..n {
            if self.n_e[i] < 0.0 {
                self.n_e[i] = 0.0;
            }
        }
    }

    /// Full solve: inject electrons as initial distribution, then evolve with cooling.
    pub fn solve(
        &mut self,
        n_bins: usize,
        gamma_max: f64,
        p_val: f64,
        gamma_m: f64,
        b: f64,
        n_total: f64,
        dt: f64,
        dln_v_dt: f64,
    ) {
        self.resize(n_bins, gamma_max);
        self.build_grid();
        self.inject_electrons(p_val, gamma_m, n_total);
        self.compute_cooling(b, dln_v_dt);
        self.compute_delta();
        self.solve_tridiagonal(dt);
    }

    // --- Public accessors for pair production module ---

    /// Build grid (public wrapper for pairs module).
    pub fn build_grid_pub(&mut self) {
        self.build_grid();
    }

    /// Get γ value at index i.
    pub fn gamma_at(&self, i: usize) -> f64 {
        self.gamma[i]
    }

    /// Add pair injection source to existing source term.
    pub fn add_pair_source(&mut self, pair_source: &[f64]) {
        let n = self.n_bins.min(pair_source.len());
        for i in 0..n {
            self.source[i] += pair_source[i];
        }
    }

    /// Public wrapper for compute_cooling.
    pub fn compute_cooling_pub(&mut self, b: f64, dln_v_dt: f64) {
        self.compute_cooling(b, dln_v_dt);
    }

    /// Public wrapper for compute_delta.
    pub fn compute_delta_pub(&mut self) {
        self.compute_delta();
    }

    /// Re-inject base electrons and add pair source, then re-solve.
    pub fn reinject_and_solve(&mut self, p_val: f64, gamma_m: f64, n_total: f64, pair_source: &[f64], dt: f64) {
        self.inject_electrons(p_val, gamma_m, n_total);
        // Add pair source to the source term (continuous injection of pairs)
        let n = self.n_bins.min(pair_source.len());
        for i in 0..n {
            self.source[i] = pair_source[i];
        }
        self.solve_tridiagonal(dt);
    }

    /// Compute synchrotron emissivity j_ν = ∫ P(ν, γ) N(γ) dγ
    /// where P(ν, γ) = (√3 e³ B / m_e c²) F(ν/ν_c(γ))
    /// and ν_c(γ) = 3eB γ² / (4π m_e c).
    pub fn emissivity(&self, nu: f64, b: f64) -> f64 {
        if b <= 0.0 || nu <= 0.0 {
            return 0.0;
        }
        let coeff = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b
            / (MASS_E * C_SPEED * C_SPEED);
        let nu_coeff = 4.0 * PI * MASS_E * C_SPEED / (3.0 * E_CHARGE * b);

        let n = self.n_bins;
        let mut j_nu = 0.0;

        for i in 0..n {
            let g = self.gamma[i];
            if g <= 1.0 || self.n_e[i] <= 0.0 {
                continue;
            }
            let nu_c = g * g / nu_coeff; // characteristic frequency for this γ
            let x = nu / nu_c;
            let f_x = synchrotron_f(x);
            j_nu += coeff * f_x * self.n_e[i] * self.d_gamma_bar[i];
        }

        j_nu
    }

    /// Compute SSA absorption coefficient α_ν.
    /// α_ν = -(1/(8π m_e ν²)) ∫ P(ν, γ) γ² d/dγ[N(γ)/γ²] dγ
    pub fn absorption(&self, nu: f64, b: f64) -> f64 {
        if b <= 0.0 || nu <= 0.0 {
            return 0.0;
        }
        let n = self.n_bins;
        if n < 3 {
            return 0.0;
        }

        let coeff = 3.0_f64.sqrt() * E_CHARGE * E_CHARGE * E_CHARGE * b
            / (MASS_E * C_SPEED * C_SPEED);
        let nu_coeff = 4.0 * PI * MASS_E * C_SPEED / (3.0 * E_CHARGE * b);
        let pre = -1.0 / (8.0 * PI * MASS_E * nu * nu);

        let mut alpha = 0.0;
        for i in 1..n - 1 {
            let g = self.gamma[i];
            if g <= 1.0 {
                continue;
            }
            let nu_c = g * g / nu_coeff;
            let x = nu / nu_c;
            let f_x = synchrotron_f(x);

            // d/dγ[N/γ²] via centered difference
            let f_plus = self.n_e[i + 1] / (self.gamma[i + 1] * self.gamma[i + 1]);
            let f_minus = self.n_e[i - 1] / (self.gamma[i - 1] * self.gamma[i - 1]);
            let dfdg = (f_plus - f_minus) / (self.gamma[i + 1] - self.gamma[i - 1]);

            alpha += pre * coeff * f_x * g * g * dfdg * self.d_gamma_bar[i];
        }

        // α must be non-negative
        alpha.max(0.0)
    }

    /// Compute intensity with SSA: j/α · (1 - exp(-α·dr)) if optically thick,
    /// else j·dr if optically thin.
    pub fn intensity_with_ssa(&self, nu: f64, b: f64, dr: f64) -> f64 {
        let j = self.emissivity(nu, b);
        let alpha = self.absorption(nu, b);

        if alpha * dr > 1e-6 {
            (j / alpha) * (1.0 - (-alpha * dr).exp())
        } else {
            j * dr
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-local solver pool for rayon parallelism
// ---------------------------------------------------------------------------

thread_local! {
    static SOLVER: RefCell<ChangCooperSolver> = RefCell::new(ChangCooperSolver::new(300, 1e8));
}

// ---------------------------------------------------------------------------
// Radiation model function
// ---------------------------------------------------------------------------

/// Numeric synchrotron model using Chang-Cooper electron distribution.
///
/// Dict parameters:
/// - `eps_e`, `eps_b`, `p` — standard microphysics
/// - `n_gamma` (optional, default 300) — number of γ bins
/// - `gamma_max` (optional, default 1e8) — maximum Lorentz factor
/// - `include_pp` (optional, default 0) — if > 0.5, include pair production iteration
pub fn sync_numeric(nu: f64, p: &Dict, blast: &Blast) -> f64 {
    let eps_e = *p.get("eps_e").expect("sync_numeric requires 'eps_e'");
    let eps_b = *p.get("eps_b").expect("sync_numeric requires 'eps_b'");
    let p_val = *p.get("p").expect("sync_numeric requires 'p'");
    let n_gamma = p.get("n_gamma").copied().unwrap_or(300.0) as usize;
    let gamma_max = p.get("gamma_max").copied().unwrap_or(1e8);
    let include_pp = p.get("include_pp").copied().unwrap_or(0.0) > 0.5;

    // Extract blast properties based on shock type
    let (b, n_blast, gamma_th, t_comv, dr) = match blast.shock_type {
        ShockType::Forward => {
            let e = blast.e_density;
            let b = (8.0 * PI * eps_b * e).sqrt();
            (b, blast.n_blast, blast.gamma_th, blast.t_comv, blast.dr)
        }
        ShockType::Reverse => {
            if blast.b3 <= 0.0 || blast.n3 <= 0.0 || blast.gamma_th3 <= 1.0 || blast.t_comv <= 0.0 {
                return 0.0;
            }
            (blast.b3, blast.n3, blast.gamma_th3, blast.t_comv, blast.dr)
        }
    };

    if b <= 0.0 || n_blast <= 0.0 || dr <= 0.0 {
        return 0.0;
    }

    // Minimum electron Lorentz factor
    let gamma_m = ((p_val - 2.0) / (p_val - 1.0) * eps_e * MASS_P / MASS_E * (gamma_th - 1.0)).max(1.0);

    // Timescale for cooling
    let dt = dr / C_SPEED;

    // Adiabatic expansion rate: dlnV/dt ~ 1/t_comv
    let dln_v_dt = if t_comv > 0.0 { 1.0 / t_comv } else { 0.0 };

    SOLVER.with(|cell| {
        let mut solver = cell.borrow_mut();
        solver.solve(n_gamma, gamma_max, p_val, gamma_m, b, n_blast, dt, dln_v_dt);

        if include_pp {
            // Pair production iteration (delegates to pairs module)
            crate::afterglow::pairs::solve_with_pairs(&mut solver, nu, b, dr, n_gamma, gamma_max, p_val, gamma_m, n_blast, dt, dln_v_dt);
        }

        solver.intensity_with_ssa(nu, b, dr)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thomas_algorithm() {
        // Solve a known tridiagonal system:
        // [2 -1  0] [x0]   [1]
        // [-1 2 -1] [x1] = [0]
        // [0 -1  2] [x2]   [1]
        // Solution: x = [1, 1, 1]
        let mut solver = ChangCooperSolver::new(3, 100.0);
        solver.a = vec![0.0, -1.0, -1.0];
        solver.b = vec![2.0, 2.0, 2.0];
        solver.c_coeff = vec![-1.0, -1.0, 0.0];
        solver.rhs = vec![1.0, 0.0, 1.0];
        solver.n_e = vec![0.0; 3];

        // Manual Thomas solve
        solver.c_prime[0] = solver.c_coeff[0] / solver.b[0];
        solver.d_prime[0] = solver.rhs[0] / solver.b[0];
        for i in 1..3 {
            let denom = solver.b[i] - solver.a[i] * solver.c_prime[i - 1];
            solver.c_prime[i] = solver.c_coeff[i] / denom;
            solver.d_prime[i] = (solver.rhs[i] - solver.a[i] * solver.d_prime[i - 1]) / denom;
        }
        solver.n_e[2] = solver.d_prime[2];
        for i in (0..2).rev() {
            solver.n_e[i] = solver.d_prime[i] - solver.c_prime[i] * solver.n_e[i + 1];
        }

        assert!((solver.n_e[0] - 1.0).abs() < 1e-10);
        assert!((solver.n_e[1] - 1.0).abs() < 1e-10);
        assert!((solver.n_e[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_synchrotron_f_peak() {
        // F(x) peaks near x ≈ 0.29
        let f_peak = synchrotron_f(0.29);
        let f_low = synchrotron_f(0.01);
        let f_high = synchrotron_f(5.0);
        assert!(f_peak > f_low, "F(0.29) = {} should be > F(0.01) = {}", f_peak, f_low);
        assert!(f_peak > f_high, "F(0.29) = {} should be > F(5.0) = {}", f_peak, f_high);
    }

    #[test]
    fn test_synchrotron_f_boundaries() {
        assert_eq!(synchrotron_f(0.0), 0.0);
        assert_eq!(synchrotron_f(-1.0), 0.0);
        assert!(synchrotron_f(50.0) < 1e-10);
    }

    #[test]
    fn test_power_law_injection() {
        let mut solver = ChangCooperSolver::new(200, 1e6);
        solver.build_grid();
        solver.inject_electrons(2.5, 10.0, 1e3);

        // N(γ) should be nonzero in the range [gamma_m, gamma_max]
        let mut has_nonzero = false;
        for i in 0..200 {
            if solver.gamma[i] >= 10.0 && solver.gamma[i] <= 1e6 {
                if solver.n_e[i] > 0.0 {
                    has_nonzero = true;
                }
            }
        }
        assert!(has_nonzero, "N(γ) should have nonzero values in injection range");

        // Check that integral ≈ n_total
        let mut integral = 0.0;
        for i in 0..200 {
            integral += solver.n_e[i] * solver.d_gamma_bar[i];
        }
        assert!((integral - 1e3).abs() / 1e3 < 0.01,
            "Integral of N(γ) = {} should be ≈ 1000", integral);
    }

    #[test]
    fn test_cooling_reduces_high_gamma() {
        let mut solver = ChangCooperSolver::new(200, 1e6);
        solver.solve(200, 1e6, 2.5, 100.0, 1.0, 1e4, 1e6, 0.0);

        // After cooling, high-γ electrons should be depleted
        let n_high: f64 = solver.n_e[180..200].iter().sum();
        let n_low: f64 = solver.n_e[10..30].iter().sum();
        // High gamma should have fewer electrons
        assert!(n_high < n_low,
            "High-γ should be depleted: n_high={}, n_low={}", n_high, n_low);
    }

    #[test]
    fn test_emissivity_positive() {
        let mut solver = ChangCooperSolver::new(200, 1e6);
        solver.solve(200, 1e6, 2.5, 100.0, 1.0, 1e4, 1e3, 0.0);

        let j = solver.emissivity(1e14, 1.0);
        assert!(j > 0.0, "Emissivity should be positive, got {}", j);
        assert!(j.is_finite(), "Emissivity should be finite");
    }

    #[test]
    fn test_absorption_nonnegative() {
        let mut solver = ChangCooperSolver::new(200, 1e6);
        solver.solve(200, 1e6, 2.5, 100.0, 1.0, 1e4, 1e3, 0.0);

        let a = solver.absorption(1e10, 1.0);
        assert!(a >= 0.0, "Absorption should be non-negative, got {}", a);
        assert!(a.is_finite(), "Absorption should be finite");
    }

    #[test]
    fn test_intensity_with_ssa() {
        let mut solver = ChangCooperSolver::new(200, 1e6);
        solver.solve(200, 1e6, 2.5, 100.0, 1.0, 1e4, 1e3, 0.0);

        let i = solver.intensity_with_ssa(1e14, 1.0, 1e15);
        assert!(i > 0.0, "Intensity should be positive, got {}", i);
        assert!(i.is_finite(), "Intensity should be finite");
    }

    #[test]
    fn test_sync_numeric_with_blast() {
        let mut p = Dict::new();
        p.insert("eps_e".into(), 0.1);
        p.insert("eps_b".into(), 0.01);
        p.insert("p".into(), 2.5);

        let blast = Blast {
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
            t_comv: 1e4,
            ..Blast::default()
        };

        let result = sync_numeric(1e14, &p, &blast);
        assert!(result > 0.0, "sync_numeric should give positive result, got {}", result);
        assert!(result.is_finite(), "sync_numeric should be finite");
    }
}
