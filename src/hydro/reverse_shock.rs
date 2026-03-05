use crate::constants::*;
use crate::hydro::tools::Tool;

// ---------------------------------------------------------------------------
// Physical constants derived from CGS fundamentals
// ---------------------------------------------------------------------------
const C2: f64 = C_SPEED * C_SPEED;
const SIGMA_CUT: f64 = 1e-6; // below this σ is treated as 0

// ---------------------------------------------------------------------------
// Smoothstep: smooth interpolation from 1 (x<=edge1) to 0 (x>=edge0)
// Matches VegasAfterglow smoothstep(edge0, edge1, x) exactly.
// ---------------------------------------------------------------------------
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = (x - edge0) / (edge1 - edge0);
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Adiabatic index for relativistic gas: γ_ad = 4/3 + 1/(3Γ)
// ---------------------------------------------------------------------------
pub fn adiabatic_idx(gamma_rel: f64) -> f64 {
    4.0 / 3.0 + 1.0 / (3.0 * gamma_rel)
}

// ---------------------------------------------------------------------------
// Reverse shock state vector (12 variables)
// ---------------------------------------------------------------------------
/// State vector for the coupled forward-reverse shock ODE system.
/// Regions: (1) unshocked ISM, (2) shocked ISM, (3) shocked ejecta, (4) unshocked ejecta.
#[derive(Clone, Debug)]
pub struct ReverseShockState {
    pub gamma: f64,       // bulk Lorentz factor of shocked region (region 2/3 contact)
    pub x4: f64,          // comoving width of unshocked ejecta (region 4)
    pub x3: f64,          // comoving width of reverse shock region (region 3)
    pub m2: f64,          // shocked ISM mass per solid angle
    pub m3: f64,          // shocked ejecta mass per solid angle
    pub u2_th: f64,       // internal thermal energy per solid angle (region 2)
    pub u3_th: f64,       // internal thermal energy per solid angle (region 3)
    pub r: f64,           // radius
    pub t_comv: f64,      // comoving time
    pub theta: f64,       // angular coordinate
    pub eps4: f64,        // energy per solid angle in unshocked ejecta (region 4)
    pub m4: f64,          // mass per solid angle in unshocked ejecta (region 4)
}

impl Default for ReverseShockState {
    fn default() -> Self {
        ReverseShockState {
            gamma: 1.0,
            x4: 0.0,
            x3: 0.0,
            m2: 0.0,
            m3: 0.0,
            u2_th: 0.0,
            u3_th: 0.0,
            r: 0.0,
            t_comv: 0.0,
            theta: 0.0,
            eps4: 0.0,
            m4: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shock jump conditions (magnetized relativistic MHD)
// ---------------------------------------------------------------------------

/// Downstream four-velocity from relativistic shock jump conditions.
/// For σ=0: u_down ≈ 3/(γ_rel + 2) (strong shock limit scaled by downstream velocity).
/// For σ>0: solves cubic polynomial via Cardano's formula.
pub fn compute_downstr_4vel(gamma_rel: f64, sigma: f64) -> f64 {
    if gamma_rel <= 1.0 {
        return 0.0;
    }
    let u_rel = (gamma_rel * gamma_rel - 1.0).sqrt();

    if sigma <= SIGMA_CUT {
        // Unmagnetized: relativistic Taub adiabat (matching VegasAfterglow exactly)
        let ad_idx = adiabatic_idx(gamma_rel);
        let gm1 = gamma_rel - 1.0;
        let am1 = ad_idx - 1.0;
        let am2 = ad_idx - 2.0;
        let denom = -ad_idx * am2 * gm1 + 2.0;
        if denom <= 0.0 {
            return u_rel / (4.0 * gamma_rel).sqrt();
        }
        let u_down_sq = gm1 * am1 * am1 / denom;
        if u_down_sq <= 0.0 {
            return 0.0;
        }
        return u_down_sq.abs().sqrt();
    }

    // Magnetized case: solve cubic a*x³ + b*x² + c*x + d = 0
    // where x = u_down² (downstream four-velocity squared)
    // From Zhang & Kobayashi (2005) magnetized jump conditions
    let g = gamma_rel;
    let s = sigma;
    let g2 = g * g;

    // Coefficients of the cubic in u_d² (approximation from VegasAfterglow)
    let a = 8.0 * s + 1.0;
    let b = -(4.0 * s * (2.0 * g2 - 1.0) + (g2 - 1.0) * (4.0 / 3.0 + 1.0 / (3.0 * g)));
    let c = 4.0 * s * s * (g2 - 1.0);
    let _d = 0.0;

    // Since d=0, factor: x * (a*x² + b*x + c) = 0
    // Non-trivial root from quadratic: a*x² + b*x + c = 0
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        // Fallback to unmagnetized
        return u_rel / (4.0 * gamma_rel).sqrt();
    }
    let x = (-b - discriminant.sqrt()) / (2.0 * a);
    if x > 0.0 {
        x.sqrt()
    } else {
        let x2 = (-b + discriminant.sqrt()) / (2.0 * a);
        if x2 > 0.0 { x2.sqrt() } else { 0.0 }
    }
}

/// Upstream four-velocity in the shock frame.
fn compute_upstr_4vel(u_down: f64, gamma_rel: f64) -> f64 {
    let g_down = (1.0 + u_down * u_down).sqrt();
    let u_rel = (gamma_rel * gamma_rel - 1.0).sqrt();
    (g_down * u_rel + u_down * gamma_rel).abs()
}

/// Compression ratio: u_upstream / u_downstream (four-velocity ratio).
pub fn compute_4vel_jump(gamma_rel: f64, sigma: f64) -> f64 {
    let u_down = compute_downstr_4vel(gamma_rel, sigma);
    if u_down == 0.0 {
        return 4.0 * gamma_rel; // non-magnetized strong shock limit
    }
    let u_up = compute_upstr_4vel(u_down, gamma_rel);
    u_up / u_down
}

/// Relative Lorentz factor between two shells.
pub fn compute_rel_gamma(gamma1: f64, gamma2: f64) -> f64 {
    let u1 = (gamma1 * gamma1 - 1.0).sqrt();
    let u2 = (gamma2 * gamma2 - 1.0).sqrt();
    let beta1 = u1 / gamma1;
    let beta2 = u2 / gamma2;
    gamma1 * gamma2 * (1.0 - beta1 * beta2)
}

/// Upstream magnetic field from magnetization parameter.
/// B₄ = √(4π σ ρ₄ c²)
pub fn compute_upstr_b(rho_up: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || rho_up <= 0.0 {
        return 0.0;
    }
    (4.0 * PI * C2 * sigma * rho_up).sqrt()
}

/// Weibel instability magnetic field: B = √(8π ε_B e_th)
pub fn compute_weibel_b(eps_b: f64, e_th: f64) -> f64 {
    if eps_b <= 0.0 || e_th <= 0.0 {
        return 0.0;
    }
    (8.0 * PI * eps_b * e_th).sqrt()
}

/// Downstream magnetic field = Weibel + compressed upstream.
pub fn compute_downstr_b(eps_b: f64, rho_up: f64, b_up: f64, gamma_th: f64, comp_ratio: f64) -> f64 {
    let rho_down = rho_up * comp_ratio;
    let e_th = (gamma_th - 1.0) * rho_down * C2;
    compute_weibel_b(eps_b, e_th) + b_up * comp_ratio
}

/// Effective Lorentz factor: Γ_eff = (α Γ² − α + 1) / Γ
fn compute_effective_gamma(ad_idx: f64, gamma: f64) -> f64 {
    (ad_idx * gamma * gamma - ad_idx + 1.0) / gamma
}

/// Derivative of effective Lorentz factor w.r.t. Γ
fn compute_effective_gamma_dgamma(ad_idx: f64, gamma: f64) -> f64 {
    let g2 = gamma * gamma;
    (ad_idx * g2 + ad_idx - 1.0) / g2
}

/// Sound speed for relativistic gas
fn compute_sound_speed(gamma_rel: f64) -> f64 {
    let ad = adiabatic_idx(gamma_rel);
    let val = ad * (ad - 1.0) * (gamma_rel - 1.0) / (1.0 + (gamma_rel - 1.0) * ad);
    val.abs().sqrt() * C_SPEED
}

/// Shell spreading rate: c_s * dt_comv/dt
/// Matches VegasAfterglow compute_shell_spreading_rate.
fn compute_shell_spreading_rate(gamma_rel: f64, dt_comv_dt: f64) -> f64 {
    let cs = compute_sound_speed(gamma_rel);
    cs * dt_comv_dt
}

/// Thermal Lorentz factor: Γ_th = U_th/(m c²) + 1
pub fn compute_gamma_th(u_th: f64, m: f64) -> f64 {
    if m <= 0.0 || u_th <= 0.0 {
        return 1.0;
    }
    u_th / (m * C2) + 1.0
}

// ---------------------------------------------------------------------------
// Forward-Reverse Shock ODE system
// ---------------------------------------------------------------------------

/// Coupled forward-reverse shock equation system.
/// Ports FRShockEqn from VegasAfterglow reverse-shock.tpp.
pub struct FRShockEqn {
    // Medium/density parameters
    pub nwind: f64,
    pub nism: f64,

    // Ejecta initial conditions
    pub gamma4: f64,         // initial ejecta Lorentz factor
    pub m4_init: f64,        // initial ejecta mass per solid angle
    pub eps4_init: f64,      // initial ejecta energy per solid angle

    // Engine duration (shell thickness)
    pub duration: f64,        // engine activity time [s] (0 = thin shell)
    pub deps0_dt: f64,        // energy injection rate from engine [erg/s/sr]
    pub dm0_dt: f64,          // mass injection rate from engine [g/s/sr]

    // Additional energy injection (beyond engine duration)
    pub t0_injection: f64,
    pub l_injection: f64,
    pub m_dot_injection: f64,

    // Magnetization
    pub sigma_init: f64,

    // Radiation parameters (for radiative efficiency)
    pub eps_e_fwd: f64,
    pub eps_b_fwd: f64,
    pub p_fwd: f64,
    pub eps_e_rs: f64,
    pub eps_b_rs: f64,
    pub p_rs: f64,

    // Cross state (saved when RS crosses ejecta)
    pub crossing_done: bool,
    pub r_x: f64,            // radius at crossing
    pub u_x: f64,            // four-velocity at crossing
    pub v3_comv_x: f64,      // comoving volume of region 3 at crossing
    pub rho3_x: f64,         // density of region 3 at crossing
    pub b3_ordered_x: f64,   // ordered B-field in region 3 at crossing

    // Tool for density computation
    tool: Tool,
}

impl FRShockEqn {
    pub fn new(
        nwind: f64,
        nism: f64,
        k: f64,
        gamma4: f64,
        m4_init: f64,
        eps4_init: f64,
        sigma_init: f64,
        eps_e_fwd: f64,
        eps_b_fwd: f64,
        p_fwd: f64,
        eps_e_rs: f64,
        eps_b_rs: f64,
        p_rs: f64,
        duration: f64,
        t0_injection: f64,
        l_injection: f64,
        m_dot_injection: f64,
        rtol: f64,
    ) -> Self {
        // Compute engine injection rates from total ejecta properties and duration.
        // Matches VegasAfterglow: deps0_dt = eps_k / T0, dm0_dt = deps0_dt / (Gamma4 * c²)
        let effective_duration = if duration > 0.0 { duration } else { 1.0 }; // avoid div by zero
        let deps0_dt = eps4_init / effective_duration;
        let dm0_dt = deps0_dt / (gamma4 * C2);

        FRShockEqn {
            nwind,
            nism,
            gamma4,
            m4_init,
            eps4_init,
            sigma_init,
            eps_e_fwd,
            eps_b_fwd,
            p_fwd,
            eps_e_rs,
            eps_b_rs,
            p_rs,
            crossing_done: false,
            r_x: 0.0,
            u_x: 0.0,
            v3_comv_x: 0.0,
            rho3_x: 0.0,
            b3_ordered_x: 0.0,
            duration,
            deps0_dt,
            dm0_dt,
            t0_injection,
            l_injection,
            m_dot_injection,
            tool: Tool::new_with_k(nwind, nism, k, rtol, 1),
        }
    }

    /// Compute comoving shell width at time t0 given engine duration T.
    /// Matches VegasAfterglow compute_init_comv_shell_width().
    fn compute_init_comv_shell_width(gamma4: f64, t0: f64, duration: f64) -> f64 {
        let beta4 = (1.0 - 1.0 / (gamma4 * gamma4)).sqrt();
        if duration <= 0.0 {
            // Thin-shell fallback: use sound-speed spreading
            let cs = compute_sound_speed(gamma4);
            return gamma4 * (cs * t0);
        }
        if t0 < duration {
            // Pure injection phase: shell width = Γ₄ β₄ c t₀
            gamma4 * t0 * beta4 * C_SPEED
        } else {
            // Injection complete + shell spreading: Γ₄ β₄ c T + c_s (t₀ - T) Γ₄
            let cs = compute_sound_speed(gamma4);
            gamma4 * duration * beta4 * C_SPEED + cs * (t0 - duration) * gamma4
        }
    }

    /// Set initial state from ejecta parameters at time t0.
    /// Matches VegasAfterglow set_init_state(): ejecta mass/energy accumulate
    /// over min(t0, duration), and shell width depends on duration.
    pub fn set_init_state(&self, t0: f64) -> ReverseShockState {
        let beta4 = (1.0 - 1.0 / (self.gamma4 * self.gamma4)).sqrt();
        let r0 = beta4 * C_SPEED * t0 / (1.0 - beta4);

        // Accumulated ejecta properties up to min(t0, duration)
        let (eps4_0, m4_0) = if self.duration > 0.0 {
            let dt = t0.min(self.duration);
            (self.deps0_dt * dt, self.dm0_dt * dt)
        } else {
            // No duration specified: use full ejecta at once (original behavior)
            (self.eps4_init, self.m4_init)
        };

        // Comoving ejecta shell width
        let x4_init = Self::compute_init_comv_shell_width(self.gamma4, t0, self.duration);

        // Enclosed ambient mass from medium density profile (general k)
        let m2_init = self.tool.solve_swept_number(r0) * MASS_P;

        // Total ejecta mass for Γ estimate (use full mass if duration specified)
        let m_jet_total = if self.duration > 0.0 {
            self.dm0_dt * self.duration
        } else {
            self.m4_init
        };

        // Initial Lorentz factor from momentum conservation
        let gamma_init = if m_jet_total <= 0.0 || m2_init < 1e-20 * m_jet_total {
            self.gamma4
        } else {
            let ratio = m2_init / m_jet_total;
            if ratio < 0.01 {
                self.gamma4 / (1.0 + ratio)
            } else {
                ((eps4_0 + m2_init * C2) / ((m4_0 + m2_init) * C2)).max(1.0)
            }
        };

        // Seed RS matching VegasAfterglow: x3 = x4*seed, m3 = m4*comp_ratio*x3/x4
        let seed = 1e-8;
        let gamma_rel_init = compute_rel_gamma(self.gamma4, gamma_init);
        let sigma_init_check = if m4_0 > 0.0 {
            let s = eps4_0 / (self.gamma4 * m4_0 * C2) - 1.0;
            if s > SIGMA_CUT { s } else { 0.0 }
        } else { 0.0 };
        let comp_ratio_init = if gamma_rel_init > 1.0 {
            compute_4vel_jump(gamma_rel_init, sigma_init_check)
        } else {
            4.0 // default strong-shock limit
        };

        let x3_init = x4_init * seed;
        let m3_init = m4_0 * comp_ratio_init * seed; // VegasAfterglow: m4*comp*x3/x4

        // Initial thermal energy from shock heating
        let u2_th_init = (gamma_init - 1.0) * m2_init * C2;
        let u3_th_init = (gamma_rel_init - 1.0) * m3_init * C2;

        ReverseShockState {
            gamma: gamma_init,
            x4: x4_init,  // full width (not reduced by seed)
            x3: x3_init,
            m2: m2_init,
            m3: m3_init,
            u2_th: u2_th_init,
            u3_th: u3_th_init,
            r: r0,
            t_comv: r0 / ((gamma_init * gamma_init - 1.0).sqrt().max(1e-30) * C_SPEED),
            theta: 0.0,
            eps4: eps4_0,  // full energy (not reduced by seed)
            m4: m4_0,      // full mass (not reduced by seed; dm4 = -dm3 + inject handles it)
        }
    }

    /// Four-velocity of unshocked ejecta: u₄ = √(Γ₄² - 1) * c
    fn u4(&self) -> f64 {
        (self.gamma4 * self.gamma4 - 1.0).sqrt() * C_SPEED
    }

    /// Magnetization parameter: σ = ε₄/(Γ₄ m₄_total c²) - 1
    /// Uses total ejecta mass (m3 + m4) to match VegasAfterglow convention,
    /// where eps4 and m4 both accumulate with injection only (no dm3 subtraction for eps4).
    pub fn compute_shell_sigma(&self, state: &ReverseShockState) -> f64 {
        let m4_total = state.m3 + state.m4; // total ejecta = VegasAfterglow's m4
        if m4_total <= 0.0 {
            return 0.0;
        }
        let sigma = state.eps4 / (self.gamma4 * m4_total * C2) - 1.0;
        if sigma > SIGMA_CUT { sigma } else { 0.0 }
    }

    /// Injection efficiency: ratio of current mass injection rate to initial rate.
    /// Matches VegasAfterglow: f = min(dm4_dt / dm0_dt, 1.0).
    /// Returns 0 when injection has stopped, 1 during full injection.
    fn injection_efficiency(&self, dm4_dt: f64) -> f64 {
        if self.dm0_dt > 0.0 && dm4_dt > 0.0 {
            (dm4_dt / self.dm0_dt).min(1.0)
        } else {
            0.0
        }
    }

    /// Check if reverse shock crossing is complete.
    /// Uses both mass ratio and shell width criteria to detect when the RS
    /// has swept through the ejecta, avoiding the x4→0 singularity.
    pub fn crossing_complete(&self, state: &ReverseShockState, t: f64) -> bool {
        // Check engine injection has stopped
        if self.duration > 0.0 {
            if smoothstep(self.duration * 1.5, self.duration * 0.5, t) > 1e-6 {
                return false;
            }
        }
        // Check additional injection has stopped
        if self.t0_injection > 0.0 {
            if smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t) > 1e-6 {
                return false;
            }
        }

        // Mass criterion: nearly all ejecta swept (matches VegasAfterglow 0.999 threshold)
        if state.m3 >= 0.999 * (state.m3 + state.m4) {
            return true;
        }

        // Shell width criterion: unshocked shell effectively gone
        // (prevents the rho4 = m4/(r²x4) → ∞ singularity)
        if state.x4 > 0.0 && state.x3 > 0.0 && state.x4 < 1e-4 * state.x3 {
            return true;
        }

        false
    }

    /// Save state at RS crossing point for post-crossing evolution
    pub fn save_cross_state(&mut self, state: &ReverseShockState) {
        self.crossing_done = true;
        self.r_x = state.r;
        self.u_x = (state.gamma * state.gamma - 1.0).sqrt();
        self.v3_comv_x = state.r * state.r * state.x3;

        let gamma34 = compute_rel_gamma(self.gamma4, state.gamma);
        let sigma = self.compute_shell_sigma(state);
        let comp_ratio = compute_4vel_jump(gamma34, sigma);

        // Upstream density: use total ejecta mass / unshocked volume
        // (matches VegasAfterglow: state.m4 / (r² x4) where m4 = total)
        let m4_total = state.m3 + state.m4;
        let rho4 = if state.x4 > 0.0 && state.r > 0.0 {
            m4_total / (state.r * state.r * state.x4)
        } else {
            0.0
        };

        self.rho3_x = rho4 * comp_ratio;
        let b4 = compute_upstr_b(rho4, sigma);
        self.b3_ordered_x = b4 * comp_ratio;
    }

    // ---- ODE right-hand-side components ----

    /// dr/dt = u*(Γ+u)*c  (matches VegasAfterglow coordinate system)
    /// This equals β c / (1-β), accounting for the lab-frame radial coordinate.
    fn compute_dr_dt(&self, state: &ReverseShockState) -> f64 {
        let u = (state.gamma * state.gamma - 1.0).sqrt();
        u * (state.gamma + u) * C_SPEED
    }

    /// dm₂/dt = ρ_ISM r² dr/dt (forward shock sweeps ISM)
    fn compute_dm2_dt(&self, state: &ReverseShockState, dr_dt: f64) -> f64 {
        let rho = self.tool.solve_density(state.r) * MASS_P;
        rho * state.r * state.r * dr_dt
    }

    /// dε₄/dt — energy injection from engine (during duration) plus additional injection
    fn compute_deps4_dt(&self, _state: &ReverseShockState, t: f64) -> f64 {
        let mut rate = 0.0;
        // Engine injection during duration (smoothstep shutdown)
        if self.duration > 0.0 && self.deps0_dt > 0.0 {
            let envelope = smoothstep(self.duration * 1.5, self.duration * 0.5, t);
            rate += self.deps0_dt * envelope;
        }
        // Additional injection beyond engine
        if self.t0_injection > 0.0 && self.l_injection > 0.0 {
            let envelope = smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t);
            rate += self.l_injection * envelope;
        }
        rate
    }

    /// dm₄/dt — mass injection from engine (during duration) plus additional injection
    fn compute_dm4_dt(&self, _state: &ReverseShockState, t: f64) -> f64 {
        let mut rate = 0.0;
        // Engine injection during duration
        if self.duration > 0.0 && self.dm0_dt > 0.0 {
            let envelope = smoothstep(self.duration * 1.5, self.duration * 0.5, t);
            rate += self.dm0_dt * envelope;
        }
        // Additional injection beyond engine
        if self.t0_injection > 0.0 && self.m_dot_injection > 0.0 {
            let envelope = smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t);
            rate += self.m_dot_injection * envelope;
        }
        rate
    }

    /// Shock heating rate: (Γ_rel - 1) c² dm/dt
    fn shock_heating_rate(gamma_rel: f64, dm_dt: f64) -> f64 {
        (gamma_rel - 1.0) * C2 * dm_dt
    }

    /// Adiabatic cooling rate for region 2.
    /// Matches VegasAfterglow: -(ad_idx - 1) * (2*dr/dt/r + dx4/dt/x4) * U2_th
    fn adiabatic_cooling_rate_2(&self, state: &ReverseShockState, dr_dt: f64, dx4_dt: f64) -> f64 {
        if state.r <= 0.0 {
            return 0.0;
        }
        let ad_idx = adiabatic_idx(state.gamma);
        let mut dlnv_dt = 2.0 * dr_dt / state.r;
        if state.x4 > 0.0 {
            dlnv_dt += dx4_dt / state.x4;
        }
        -(ad_idx - 1.0) * dlnv_dt * state.u2_th
    }

    /// Adiabatic cooling rate for region 3.
    /// Matches VegasAfterglow: -(ad_idx - 1) * (2*dr/dt/r + dx3/dt/x3) * U3_th
    fn adiabatic_cooling_rate_3(&self, state: &ReverseShockState, dr_dt: f64, dx3_dt: f64, gamma34: f64) -> f64 {
        if state.r <= 0.0 {
            return 0.0;
        }
        let ad_idx = adiabatic_idx(gamma34);
        let mut dlnv_dt = 2.0 * dr_dt / state.r;
        if state.x3 > 0.0 {
            dlnv_dt += dx3_dt / state.x3;
        }
        -(ad_idx - 1.0) * dlnv_dt * state.u3_th
    }

    /// Radiative efficiency for forward shock
    fn radiative_efficiency_fwd(&self, state: &ReverseShockState) -> f64 {
        let gamma_th2 = compute_gamma_th(state.u2_th, state.m2);
        self.compute_eps_rad(gamma_th2, state.t_comv, state.gamma,
                            self.eps_e_fwd, self.eps_b_fwd, self.p_fwd)
    }

    /// Compute radiative efficiency
    fn compute_eps_rad(&self, gamma_th: f64, t_comv: f64, gamma: f64,
                       eps_e: f64, eps_b: f64, p: f64) -> f64 {
        if gamma_th <= 1.0 || t_comv <= 0.0 {
            return 0.0;
        }
        let gamma_m = (p - 2.0) / (p - 1.0) * eps_e * (gamma_th - 1.0) * MASS_P / MASS_E + 1.0;
        let u = (gamma * gamma - 1.0).sqrt();
        let e_th = (gamma_th - 1.0) * MASS_P * C2; // per proton
        let b = compute_weibel_b(eps_b, e_th * 4.0 * gamma * self.tool.solve_density(1e17)); // approximate
        if b <= 0.0 {
            return 0.0;
        }
        let gamma_c = (6.0 * PI * MASS_E * C_SPEED / (SIGMA_T * b * b * t_comv)).max(1.0);
        let ratio = (gamma_m / gamma_c).abs();
        if ratio < 1.0 && p > 2.0 {
            if ratio < 1e-2 { return 0.0; }
            eps_e * ratio.powf(p - 2.0)
        } else {
            eps_e
        }
    }

    /// dx₄/dt — unshocked ejecta width evolution.
    /// Matches VegasAfterglow: during injection x4 grows at u4 speed,
    /// after injection x4 grows by sound-speed spreading.
    fn compute_dx4_dt(&self, state: &ReverseShockState, dt_comv_dt: f64, dm4_dt: f64) -> f64 {
        let spreading = compute_shell_spreading_rate(self.gamma4, dt_comv_dt);
        let f = self.injection_efficiency(dm4_dt);
        if f > 1e-6 {
            f * self.u4() + (1.0 - f) * spreading
        } else {
            spreading
        }
    }

    /// dx₃/dt — reverse shock region width evolution.
    /// Matches VegasAfterglow: blends relativistic crossing velocity with
    /// sound-speed spreading, weighted by injection efficiency.
    /// Note: our m4 = unshocked mass, VegasAfterglow's m4 = total ejecta.
    fn compute_dx3_dt(&self, state: &ReverseShockState, dt_comv_dt: f64,
                      gamma34: f64, _sigma: f64, comp_ratio: f64, dm4_dt: f64) -> f64 {
        let spreading = compute_shell_spreading_rate(gamma34, dt_comv_dt);

        // m4_total = m3 + m4 (our convention: m4 is unshocked only)
        let m4_total = state.m3 + state.m4;
        if m4_total <= 0.0 {
            return spreading;
        }

        // Blend crossing/spreading using injection efficiency and remaining mass
        let f = self.injection_efficiency(dm4_dt);
        let remaining = state.m4; // unshocked mass (= VegasAfterglow's m4 - m3)
        let crossing_w = f + (1.0 - f) * remaining / m4_total;

        if crossing_w < 1e-6 {
            return spreading;
        }

        // Relativistic RS crossing velocity (VegasAfterglow formula)
        let beta3 = (1.0 - 1.0 / (state.gamma * state.gamma)).sqrt();
        let beta4 = (1.0 - 1.0 / (self.gamma4 * self.gamma4)).sqrt();
        let denom = (1.0 - beta3) * (state.gamma * comp_ratio / self.gamma4 - 1.0);
        if denom.abs() < 1e-30 {
            return spreading;
        }
        let dx3dt = (beta4 - beta3) * C_SPEED / denom;
        let crossing = (dx3dt * state.gamma).abs();

        crossing_w * crossing + (1.0 - crossing_w) * spreading
    }

    /// dm₃/dt — mass accumulation in reverse shock.
    /// Matches VegasAfterglow: dm3 = column_density * dx3_dt
    /// where column_density = eff_mass * comp_ratio / x4.
    /// Note: our m4 = unshocked mass, VegasAfterglow's m4 = total ejecta.
    fn compute_dm3_dt(&self, state: &ReverseShockState, dx3_dt: f64,
                      _gamma34: f64, _sigma: f64, comp_ratio: f64, dm4_dt: f64) -> f64 {
        let m4_total = state.m3 + state.m4;
        if m4_total <= 0.0 {
            return 0.0;
        }
        if state.x4 <= 0.0 {
            return 0.0;
        }

        let f = self.injection_efficiency(dm4_dt);
        let remaining = state.m4; // unshocked (= VegasAfterglow's m4 - m3)

        if remaining <= 0.0 && f < 1e-6 {
            return 0.0;
        }

        // Blend column density: m4_total during injection, remaining after
        let eff_mass = f * m4_total + (1.0 - f) * remaining;

        let column_den3 = eff_mass * comp_ratio / state.x4;
        let dm3dt = column_den3 * dx3_dt;

        // During active injection, cap the rate so dm3 <= dm4
        if f > 1e-6 {
            let ratio = state.m3 / m4_total;
            let cap_w = smoothstep(0.0, 1.0, ratio);
            let capped_rate = dm3dt.min(dm4_dt);
            (1.0 - cap_w) * dm3dt + cap_w * capped_rate
        } else {
            dm3dt
        }
    }

    /// dU₂/dt — thermal energy evolution in region 2 (forward shock).
    /// Matches VegasAfterglow: (1-eps_rad)*heating + adiabatic_cooling
    fn compute_du2_dt(&self, state: &ReverseShockState, dm2_dt: f64, dr_dt: f64, dx4_dt: f64) -> f64 {
        let eps_rad = self.radiative_efficiency_fwd(state);

        // Heating from sweeping ISM
        let heating = (1.0 - eps_rad) * Self::shock_heating_rate(state.gamma, dm2_dt);

        // Adiabatic cooling (already negative from the function)
        let cooling = self.adiabatic_cooling_rate_2(state, dr_dt, dx4_dt);

        heating + cooling
    }

    /// dU₃/dt — thermal energy evolution in region 3 (reverse shock).
    /// Matches VegasAfterglow: heating + adiabatic_cooling
    fn compute_du3_dt(&self, state: &ReverseShockState, dm3_dt: f64, dr_dt: f64, dx3_dt: f64,
                      gamma34: f64) -> f64 {
        // Heating from reverse shock
        let heating = Self::shock_heating_rate(gamma34, dm3_dt);

        // Adiabatic cooling (already negative from the function)
        let cooling = self.adiabatic_cooling_rate_3(state, dr_dt, dx3_dt, gamma34);

        heating + cooling
    }

    /// dΓ/dt — bulk Lorentz factor evolution from energy-momentum conservation.
    /// Matches VegasAfterglow compute_dGamma_dt exactly:
    ///   a = (Γ-1)c² dm2 + (Γ-Γ4)c² dm3 + Γ_eff2 dU2 + Γ_eff3 dU3 - deps_dt
    ///   b = (m2+m3)c² + dΓ_eff2/dΓ U2 + dΓ_eff3/dΓ U3
    ///   dΓ/dt = -a/b
    fn compute_dgamma_dt(&self, state: &ReverseShockState, dm2_dt: f64,
                         dm3_dt: f64, du2_dt: f64, du3_dt: f64,
                         deps4_residual: f64, gamma34: f64) -> f64 {
        let g = state.gamma;

        // Effective Lorentz factors (using ad_idx of contact Gamma for region 2,
        // ad_idx of Gamma34 for region 3 — matching VegasAfterglow)
        let ad2 = adiabatic_idx(g);
        let ad3 = adiabatic_idx(gamma34);
        let e_eff2 = compute_effective_gamma(ad2, g);
        let e_eff2_dg = compute_effective_gamma_dgamma(ad2, g);
        let e_eff3 = compute_effective_gamma(ad3, g);
        let e_eff3_dg = compute_effective_gamma_dgamma(ad3, g);

        // Numerator: matches VegasAfterglow exactly
        let a = (g - 1.0) * C2 * dm2_dt
              + (g - self.gamma4) * C2 * dm3_dt
              + e_eff2 * du2_dt
              + e_eff3 * du3_dt
              - deps4_residual;

        // Denominator: (m2+m3)c² + dΓ_eff/dΓ * U_th terms
        let b = (state.m2 + state.m3) * C2
              + e_eff2_dg * state.u2_th
              + e_eff3_dg * state.u3_th;

        if b.abs() < 1e-60 || (-a / b).is_nan() || (-a / b).is_infinite() {
            return 0.0;
        }

        -a / b
    }

    /// Compute all derivatives (main ODE right-hand side).
    /// Matches VegasAfterglow operator() evaluation order.
    pub fn derivatives(&self, t: f64, state: &ReverseShockState) -> ReverseShockState {
        let g = state.gamma;
        let u3 = (g * g - 1.0).sqrt();

        let dr_dt = self.compute_dr_dt(state);
        let dt_comv_dt = g + u3; // dt_comv/dt = Γ + u

        let dm2_dt = self.compute_dm2_dt(state, dr_dt);
        let deps4_dt = self.compute_deps4_dt(state, t);
        let dm4_dt_inject = self.compute_dm4_dt(state, t);

        // Relative Lorentz factor between ejecta and shocked region
        let gamma34 = compute_rel_gamma(self.gamma4, g);
        let sigma = self.compute_shell_sigma(state);
        let comp_ratio = compute_4vel_jump(gamma34, sigma);

        let dx4_dt = self.compute_dx4_dt(state, dt_comv_dt, dm4_dt_inject);
        let dx3_dt = self.compute_dx3_dt(state, dt_comv_dt, gamma34, sigma, comp_ratio, dm4_dt_inject);
        let dm3_dt = self.compute_dm3_dt(state, dx3_dt, gamma34, sigma, comp_ratio, dm4_dt_inject);

        let du2_dt = self.compute_du2_dt(state, dm2_dt, dr_dt, dx4_dt);
        let du3_dt = self.compute_du3_dt(state, dm3_dt, dr_dt, dx3_dt, gamma34);

        // Additional injection residual for gamma equation
        let deps4_residual = if self.t0_injection > 0.0 {
            let envelope = smoothstep(self.t0_injection * 1.5, self.t0_injection * 0.5, t);
            -self.l_injection * envelope + self.gamma4 * C2 * self.m_dot_injection * envelope
        } else {
            0.0
        };

        let dgamma_dt = self.compute_dgamma_dt(
            state, dm2_dt, dm3_dt, du2_dt, du3_dt,
            deps4_residual, gamma34,
        );

        ReverseShockState {
            gamma: dgamma_dt,
            x4: dx4_dt,
            x3: dx3_dt,
            m2: dm2_dt,
            m3: dm3_dt,
            u2_th: du2_dt,
            u3_th: du3_dt,
            r: dr_dt,
            t_comv: dt_comv_dt,
            theta: 0.0,
            eps4: deps4_dt,
            m4: -dm3_dt + dm4_dt_inject, // mass leaving region 4 → region 3
        }
    }

    /// RK45 adaptive step for the reverse shock ODE.
    /// Returns (new_state, new_dt, succeeded).
    pub fn step_rk45(
        &self,
        t: f64,
        state: &ReverseShockState,
        dt: f64,
        rtol: f64,
    ) -> (ReverseShockState, f64, bool) {
        // Butcher tableau (Dormand-Prince RK45)
        let k1 = self.derivatives(t, state);
        let s2 = self.add_states(state, &self.scale_state(&k1, dt * 2.0 / 9.0));
        let k2 = self.derivatives(t + dt * 2.0 / 9.0, &s2);

        let s3 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt / 12.0),
            &self.scale_state(&k2, dt / 4.0),
        ));
        let k3 = self.derivatives(t + dt / 3.0, &s3);

        let s4 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 69.0 / 128.0),
            &self.add_states(
                &self.scale_state(&k2, dt * -243.0 / 128.0),
                &self.scale_state(&k3, dt * 135.0 / 64.0),
            ),
        ));
        let k4 = self.derivatives(t + dt * 13.0 / 24.0, &s4);

        let s5 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * -17.0 / 12.0),
            &self.add_states(
                &self.scale_state(&k2, dt * 27.0 / 4.0),
                &self.add_states(
                    &self.scale_state(&k3, dt * -27.0 / 5.0),
                    &self.scale_state(&k4, dt * 16.0 / 15.0),
                ),
            ),
        ));
        let k5 = self.derivatives(t + dt, &s5);

        let s6 = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 65.0 / 432.0),
            &self.add_states(
                &self.scale_state(&k2, dt * -5.0 / 16.0),
                &self.add_states(
                    &self.scale_state(&k3, dt * 13.0 / 16.0),
                    &self.add_states(
                        &self.scale_state(&k4, dt * 4.0 / 27.0),
                        &self.scale_state(&k5, dt * 5.0 / 144.0),
                    ),
                ),
            ),
        ));
        let k6 = self.derivatives(t + dt, &s6);

        // 5th-order solution
        let result = self.add_states(state, &self.add_states(
            &self.scale_state(&k1, dt * 47.0 / 450.0),
            &self.add_states(
                &self.scale_state(&k3, dt * 12.0 / 25.0),
                &self.add_states(
                    &self.scale_state(&k4, dt * 32.0 / 225.0),
                    &self.add_states(
                        &self.scale_state(&k5, dt / 30.0),
                        &self.scale_state(&k6, dt * 6.0 / 25.0),
                    ),
                ),
            ),
        ));

        // Error estimate (difference between 4th and 5th order)
        let err = self.add_states(
            &self.scale_state(&k1, dt / 150.0),
            &self.add_states(
                &self.scale_state(&k3, dt * -3.0 / 100.0),
                &self.add_states(
                    &self.scale_state(&k4, dt * 16.0 / 75.0),
                    &self.add_states(
                        &self.scale_state(&k5, dt / 20.0),
                        &self.scale_state(&k6, dt * -6.0 / 25.0),
                    ),
                ),
            ),
        );

        let rerror = self.max_scaled_error(&err, &result, rtol);
        let succeeded = rerror < 1.0; // scaled error < 1 means within tolerance

        let boost = (0.9 * (1.0 / rerror.max(1e-30)).powf(0.2)).min(5.0).max(0.2);
        let new_dt = dt * boost;

        if succeeded {
            (result, new_dt, true)
        } else {
            (state.clone(), new_dt, false)
        }
    }

    // ---- State arithmetic helpers ----

    fn scale_state(&self, s: &ReverseShockState, factor: f64) -> ReverseShockState {
        ReverseShockState {
            gamma: s.gamma * factor,
            x4: s.x4 * factor,
            x3: s.x3 * factor,
            m2: s.m2 * factor,
            m3: s.m3 * factor,
            u2_th: s.u2_th * factor,
            u3_th: s.u3_th * factor,
            r: s.r * factor,
            t_comv: s.t_comv * factor,
            theta: s.theta * factor,
            eps4: s.eps4 * factor,
            m4: s.m4 * factor,
        }
    }

    fn add_states(&self, a: &ReverseShockState, b: &ReverseShockState) -> ReverseShockState {
        ReverseShockState {
            gamma: a.gamma + b.gamma,
            x4: a.x4 + b.x4,
            x3: a.x3 + b.x3,
            m2: a.m2 + b.m2,
            m3: a.m3 + b.m3,
            u2_th: a.u2_th + b.u2_th,
            u3_th: a.u3_th + b.u3_th,
            r: a.r + b.r,
            t_comv: a.t_comv + b.t_comv,
            theta: a.theta + b.theta,
            eps4: a.eps4 + b.eps4,
            m4: a.m4 + b.m4,
        }
    }

    /// Error norm using mixed absolute+relative tolerance (Hairer-Wanner style):
    ///   err_i / (atol + rtol * |y_i|)
    /// This prevents near-zero variables (e.g., m3 at seed value) from
    /// dominating the error estimate and forcing needlessly small steps.
    fn max_scaled_error(&self, err: &ReverseShockState, state: &ReverseShockState, rtol: f64) -> f64 {
        // Absolute tolerance: fraction of the dominant scale (ejecta energy)
        let atol = self.eps4_init * 1e-12;
        let sc = |e: f64, s: f64| -> f64 {
            e.abs() / (atol + rtol * s.abs()).max(1e-100)
        };
        sc(err.gamma, state.gamma)
            .max(sc(err.r, state.r))
            .max(sc(err.m2, state.m2))
            .max(sc(err.m3, state.m3))
            .max(sc(err.u2_th, state.u2_th))
            .max(sc(err.u3_th, state.u3_th))
            .max(sc(err.x3, state.x3))
            .max(sc(err.m4, state.m4))
    }
}

// ---------------------------------------------------------------------------
// Reverse shock ODE solver
// ---------------------------------------------------------------------------

/// Solve the coupled forward-reverse shock ODE from tmin to tmax.
/// Returns (ts, forward_ys, reverse_ys) where:
/// - forward_ys[var][itheta][it] are the forward shock primitive variables
/// - reverse_ys[var][it] are the reverse shock state variables
///
/// reverse_ys layout: [0]=Gamma, [1]=r_rs, [2]=m3, [3]=x3, [4]=u3_th,
///                    [5]=t_comv, [6]=gamma_th3, [7]=b3, [8]=n3,
///                    [9]=gamma34, [10]=n4
pub const NVAR_RS: usize = 14;

/// Solve the reverse shock ODE for a single theta cell.
/// Returns (rs_vars[NVAR_RS][nt], crossing_idx) for each saved timestep.
/// Maximum number of ODE steps before giving up (safety limit).
const MAX_ODE_STEPS: usize = 1_000_000;

/// The ODE is integrated in **retarded time** (t_ret = t_lab - R/c),
/// matching VegasAfterglow's coordinate system exactly. The save_times
/// are in lab time; the solver maps via t_lab = t_ret + r/c.
pub fn solve_reverse_shock_cell(
    eqn: &mut FRShockEqn,
    tmin: f64,       // lab time [s]
    _tmax: f64,      // lab time [s] (unused; solver fills all save_times)
    rtol: f64,
    save_times: &[f64],  // lab times [s]
) -> (Vec<Vec<f64>>, usize) {
    // Convert lab time → retarded time: t_ret = t_lab * (1 - beta4)
    let beta4 = (1.0 - 1.0 / (eqn.gamma4 * eqn.gamma4)).sqrt();
    let t_ret = tmin * (1.0 - beta4);

    // Initialise state in retarded time (matching VegasAfterglow)
    let state0 = eqn.set_init_state(t_ret);
    let mut state = state0;
    let mut t = t_ret;                                     // ODE variable
    let mut t_lab = t + state.r / C_SPEED;                 // lab time

    // Initial step size (Hairer-Wanner)
    let f0 = eqn.derivatives(t, &state);
    let d0 = state_norm(&state);
    let d1 = state_norm(&f0);
    let mut dt = if d0 < 1e-5 || d1 < 1e-5 {
        t * 1e-6
    } else {
        0.01 * d0 / d1
    };
    dt = dt.min(t * 0.1).max(t * 1e-10);

    // Output arrays
    let mut rs_out: Vec<Vec<f64>> = vec![Vec::new(); NVAR_RS];
    let mut save_idx = 0;
    let mut crossing_idx = save_times.len();
    let mut n_steps: usize = 0;

    // Save initial state for any save_times <= initial lab time
    while save_idx < save_times.len() && save_times[save_idx] <= t_lab + 1e-10 {
        save_rs_state(&state, eqn, &mut rs_out);
        save_idx += 1;
    }

    while save_idx < save_times.len() {
        if n_steps >= MAX_ODE_STEPS { break; }
        // Bail early if no progress after 50K steps (ultra-relativistic stiffness)
        if n_steps >= 50_000 && save_idx == 0 { break; }

        // Estimate retarded-time step to reach the next lab-time save point.
        // dt_lab/dt_ret = 1 + (dr/dt_ret)/c = 1 + u*(G+u), so
        // dt_ret ≈ dt_lab / (1 + u*(G+u)).
        let u = (state.gamma * state.gamma - 1.0).sqrt();
        let stretch = 1.0 + u * (state.gamma + u);
        let dt_lab_gap = save_times[save_idx] - t_lab;
        let dt_max = (dt_lab_gap / stretch).max(t * 1e-10);
        dt = dt.min(dt_max);

        let (new_state, new_dt, succeeded) = eqn.step_rk45(t, &state, dt, rtol);
        n_steps += 1;

        let dt_min = t * 1e-12;
        if dt < dt_min && !succeeded {
            t += dt_min;
            dt = dt_min * 10.0;
            t_lab = t + state.r / C_SPEED;
            continue;
        }

        if succeeded {
            t += dt;
            state = new_state;

            // Physical bounds
            state.gamma = state.gamma.max(1.0);
            state.m2 = state.m2.max(0.0);
            state.m3 = state.m3.max(0.0);
            state.m4 = state.m4.max(0.0);
            state.u2_th = state.u2_th.max(0.0);
            state.u3_th = state.u3_th.max(0.0);
            state.x3 = state.x3.max(0.0);
            state.x4 = state.x4.max(0.0);
            state.r = state.r.max(0.0);

            t_lab = t + state.r / C_SPEED;

            // Check crossing
            if !eqn.crossing_done && eqn.crossing_complete(&state, t) {
                eqn.save_cross_state(&state);
                crossing_idx = save_idx;
            }

            // Save at requested lab times
            while save_idx < save_times.len() && save_times[save_idx] <= t_lab + 1e-10 {
                save_rs_state(&state, eqn, &mut rs_out);
                save_idx += 1;
            }

            dt = new_dt;
        } else {
            dt = new_dt;
        }
    }


    // Fill remaining times with last state (for safety)
    while save_idx < save_times.len() {
        save_rs_state(&state, eqn, &mut rs_out);
        save_idx += 1;
    }

    (rs_out, crossing_idx)
}

/// Euclidean norm of a state vector (for initial step estimation)
fn state_norm(s: &ReverseShockState) -> f64 {
    (s.gamma * s.gamma + s.r * s.r + s.m2 * s.m2 + s.m3 * s.m3
     + s.u2_th * s.u2_th + s.u3_th * s.u3_th + s.x3 * s.x3
     + s.x4 * s.x4 + s.eps4 * s.eps4 + s.m4 * s.m4).sqrt()
}

/// Save RS state variables to output arrays.
fn save_rs_state(
    state: &ReverseShockState,
    eqn: &FRShockEqn,
    rs_out: &mut [Vec<f64>],
) {
    let gamma_th3 = compute_gamma_th(state.u3_th, state.m3);

    // Compute B-field in region 3
    let gamma34 = compute_rel_gamma(eqn.gamma4, state.gamma);
    let sigma = eqn.compute_shell_sigma(state);
    let comp_ratio = compute_4vel_jump(gamma34, sigma);

    // Use total ejecta mass for rho4, matching VegasAfterglow convention
    let m4_total = state.m3 + state.m4;
    let rho4 = if state.x4 > 0.0 && state.r > 0.0 {
        m4_total / (state.r * state.r * state.x4)
    } else {
        0.0
    };

    let b3 = if eqn.crossing_done {
        // Post-crossing: B-field evolves with adiabatic expansion.
        // Use compute_downstr_b with crossing-time density/B and volume compression ratio.
        // comp_ratio = V3_comv_x / V3_comv (< 1 as shell expands → B decays).
        // Matches VegasAfterglow: compute_downstr_B(eps_B, rho3_x, B3_ordered_x, Gamma3_th, comp_ratio).
        let v3_comv = state.r * state.r * state.x3;
        if v3_comv > 0.0 && eqn.v3_comv_x > 0.0 {
            let vol_ratio = eqn.v3_comv_x / v3_comv;
            compute_downstr_b(eqn.eps_b_rs, eqn.rho3_x, eqn.b3_ordered_x, gamma_th3, vol_ratio)
        } else {
            0.0
        }
    } else {
        let b4 = compute_upstr_b(rho4, sigma);
        compute_downstr_b(eqn.eps_b_rs, rho4, b4, gamma_th3, comp_ratio)
    };

    // Number density in region 3
    let n3 = if state.x3 > 0.0 && state.r > 0.0 {
        state.m3 / (state.r * state.r * state.x3 * MASS_P)
    } else {
        0.0
    };

    rs_out[0].push(state.gamma);
    rs_out[1].push(state.r);       // r_rs (same as forward r in thin shell)
    rs_out[2].push(state.m3);
    rs_out[3].push(state.x3);
    rs_out[4].push(state.u3_th);
    rs_out[5].push(state.t_comv);
    rs_out[6].push(gamma_th3);
    rs_out[7].push(b3);
    rs_out[8].push(n3);
    rs_out[9].push(gamma34);
    let n4 = if rho4 > 0.0 { rho4 / MASS_P } else { 0.0 };
    rs_out[10].push(n4);
    rs_out[11].push(state.m2);     // FS swept mass per steradian
    rs_out[12].push(state.u2_th);  // FS thermal energy per steradian
    rs_out[13].push(0.0);          // placeholder for t_crossing (filled in sim_box)
}

/// Power-law back-extrapolation for early-time zeros in RS data.
pub fn reverse_shock_early_extrap(rs_data: &mut [Vec<f64>], first_nonzero: usize) {
    if first_nonzero <= 1 || first_nonzero >= rs_data[0].len() {
        return;
    }

    // Find two reference points
    let i1 = first_nonzero;
    let i2 = (first_nonzero + 1).min(rs_data[0].len() - 1);

    for var in 0..rs_data.len() {
        let v1 = rs_data[var][i1];
        let v2 = rs_data[var][i2];
        if v1 <= 0.0 || v2 <= 0.0 || v1 == v2 {
            // Fill with first nonzero value
            for j in 0..first_nonzero {
                rs_data[var][j] = v1;
            }
        } else {
            // Log-log extrapolation
            let log_ratio = (v2 / v1).ln();
            let log_t_ratio = ((i2 as f64 + 1.0) / (i1 as f64 + 1.0)).ln();
            let slope = if log_t_ratio.abs() > 1e-30 { log_ratio / log_t_ratio } else { 0.0 };
            for j in 0..first_nonzero {
                let log_val = v1.ln() + slope * ((j as f64 + 1.0) / (i1 as f64 + 1.0)).ln();
                rs_data[var][j] = log_val.exp();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(1.0, 0.0, -0.5) - 1.0).abs() < 1e-10);
        assert!((smoothstep(1.0, 0.0, 1.5) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0, 0.0, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adiabatic_idx() {
        // Ultra-relativistic: γ = 4/3 + 1/(3*large) ≈ 4/3
        assert!((adiabatic_idx(100.0) - 4.0 / 3.0).abs() < 0.01);
        // Non-relativistic: γ = 4/3 + 1/3 = 5/3
        assert!((adiabatic_idx(1.0) - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rel_gamma() {
        // Same frame: Γ_rel = 1
        assert!((compute_rel_gamma(10.0, 10.0) - 1.0).abs() < 0.01);
        // One at rest: Γ_rel = other's Γ
        assert!((compute_rel_gamma(10.0, 1.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_compression_ratio_unmagnetized() {
        // Strong shock limit: compression ratio → 4Γ
        let gamma_rel = 10.0;
        let comp = compute_4vel_jump(gamma_rel, 0.0);
        assert!(comp > 1.0);
        assert!(comp.is_finite());
    }

    #[test]
    fn test_compression_ratio_magnetized() {
        let gamma_rel = 10.0;
        let comp_unmag = compute_4vel_jump(gamma_rel, 0.0);
        let comp_mag = compute_4vel_jump(gamma_rel, 0.1);
        // Magnetization should modify the compression ratio
        assert!(comp_mag.is_finite());
        assert!(comp_mag > 0.0);
        // With magnetization, compression ratio differs from unmagnetized
        assert!((comp_mag - comp_unmag).abs() > 0.01 || comp_mag > 0.0);
    }

    #[test]
    fn test_upstream_b() {
        let b = compute_upstr_b(1e-24, 0.1);
        assert!(b > 0.0);
        assert!(b.is_finite());
        // Zero sigma → zero B
        assert_eq!(compute_upstr_b(1e-24, 0.0), 0.0);
    }

    #[test]
    fn test_gamma_th() {
        // No thermal energy → Γ_th = 1
        assert!((compute_gamma_th(0.0, 1.0) - 1.0).abs() < 1e-10);
        // Positive thermal energy → Γ_th > 1
        let g = compute_gamma_th(1e50, 1e30);
        assert!(g > 1.0);
    }

    #[test]
    fn test_init_state() {
        let eqn = FRShockEqn::new(
            0.0, 1.0,    // ISM only
            2.0,          // k = 2 (wind profile exponent)
            100.0,        // Γ₄ = 100
            1e-5,         // m4
            1e48,         // eps4
            0.0,          // σ = 0
            0.1, 0.01, 2.3,  // FS rad params
            0.1, 0.01, 2.3,  // RS rad params
            0.0,              // duration (thin shell)
            0.0, 0.0, 0.0,   // no injection
            1e-6,
        );
        let state = eqn.set_init_state(10.0);
        assert!(state.r > 0.0);
        assert!(state.gamma >= 1.0);
        assert!(state.m2 >= 0.0);
        assert!(state.m3 > 0.0);
        assert!(state.m4 > 0.0);
    }

    #[test]
    fn test_derivatives_finite() {
        let eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, 100.0, 1e-5, 1e48, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 0.0, 1e-6,
        );
        let state = eqn.set_init_state(10.0);
        let deriv = eqn.derivatives(10.0, &state);
        assert!(deriv.r.is_finite());
        assert!(deriv.gamma.is_finite());
        assert!(deriv.m2.is_finite());
        assert!(deriv.m3.is_finite());
    }

    #[test]
    fn test_rk45_step() {
        let eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, 100.0, 1e-5, 1e48, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 0.0, 1e-6,
        );
        let state = eqn.set_init_state(10.0);
        let (new_state, new_dt, succeeded) = eqn.step_rk45(10.0, &state, 1.0, 1e-4);
        assert!(new_dt > 0.0);
        // Should either succeed or give a new dt to try
        if succeeded {
            assert!(new_state.r > state.r);
            assert!(new_state.gamma >= 1.0);
        }
    }

    #[test]
    fn test_solve_rs_cell() {
        // Use physically consistent parameters:
        // Γ₄=100, mej per sr ≈ E/(Γ c²) ≈ 1e52/(4π*100*c²) ≈ 3e26
        let gamma4 = 100.0;
        let mej = 1e52 / (4.0 * PI * gamma4 * C2);
        let eps4 = gamma4 * mej * C2; // total energy = Γ m c²
        let mut eqn = FRShockEqn::new(
            0.0, 1.0, 2.0, gamma4, mej, eps4, 0.0,
            0.1, 0.01, 2.3, 0.1, 0.01, 2.3,
            0.0, 0.0, 0.0, 0.0, 1e-3,
        );
        // Save at a few log-spaced times
        let save_times: Vec<f64> = vec![10.0, 30.0, 100.0, 300.0, 1000.0];
        let (rs_out, _crossing_idx) = solve_reverse_shock_cell(
            &mut eqn, 10.0, 1000.0, 1e-3, &save_times,
        );
        assert_eq!(rs_out.len(), NVAR_RS);
        for v in &rs_out {
            assert_eq!(v.len(), save_times.len(), "each var should have {} entries", save_times.len());
        }
        // Gamma should stay >= 1
        for &g in &rs_out[0] {
            assert!(g >= 1.0, "Gamma = {} should be >= 1", g);
        }
        // Radius should be positive
        for &r in &rs_out[1] {
            assert!(r > 0.0);
        }
    }
}
