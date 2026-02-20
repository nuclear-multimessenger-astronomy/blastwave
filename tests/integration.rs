//! Integration test: full top-hat jet simulation pipeline.
//!
//! Mirrors the Python `FluxDensity_tophat` workflow entirely in Rust:
//! build initial conditions, solve the PDE, interpolate, and compute
//! afterglow luminosity.

use std::collections::HashMap;

use jetsimpy_rs::constants::*;
use jetsimpy_rs::hydro::config::JetConfig;
use jetsimpy_rs::hydro::sim_box::SimBox;
use jetsimpy_rs::hydro::interpolate::Interpolator;
use jetsimpy_rs::hydro::tools::Tool;
use jetsimpy_rs::afterglow::eats::EATS;
use jetsimpy_rs::afterglow::afterglow::Afterglow;

/// Build a top-hat jet config matching the Python quick-start example.
fn tophat_config() -> JetConfig {
    let theta_c: f64 = 0.1;
    let eiso: f64 = 1e52;
    let lf0: f64 = 300.0;
    let n0: f64 = 1.0;
    let nwind: f64 = 0.0;
    let tmin: f64 = 10.0;
    let tmax: f64 = 1e10;

    // Build ForwardJetRes grid (arcsinh spacing), 129 edges
    let npoints = 129;
    let mut theta_edge = vec![0.0; npoints];
    let arcsinh_max = (PI / theta_c).asinh();
    for i in 0..npoints {
        theta_edge[i] = (i as f64 / (npoints - 1) as f64 * arcsinh_max).sinh() * theta_c;
    }
    theta_edge[0] = 0.0;
    theta_edge[npoints - 1] = PI;

    // Cell centers
    let ncells = npoints - 1;
    let theta: Vec<f64> = (0..ncells)
        .map(|i| (theta_edge[i] + theta_edge[i + 1]) / 2.0)
        .collect();

    // Top-hat profile sampled on a fine grid, then interpolated to cell centers
    let nfine = 10000;
    let theta_fine: Vec<f64> = (0..nfine).map(|i| i as f64 / (nfine - 1) as f64 * PI).collect();
    let mut energy_fine = vec![eiso; nfine];
    let mut lf_fine = vec![1.0f64; nfine];
    for i in 0..nfine {
        if theta_fine[i] > theta_c {
            energy_fine[i] = 0.0;
        } else {
            lf_fine[i] = lf0;
        }
    }

    // Apply isotropic tail
    let max_e = energy_fine.iter().cloned().fold(0.0f64, f64::max);
    for e in energy_fine.iter_mut() {
        if *e <= max_e * 1e-12 {
            *e = max_e * 1e-12;
        }
    }
    for lf in lf_fine.iter_mut() {
        if *lf <= 1.005 {
            *lf = 1.005;
        }
    }

    // Interpolate to cell centers
    let energy_interp = interp(&theta, &theta_fine, &energy_fine.iter().map(|e| e / 4.0 / PI / C_SPEED / C_SPEED).collect::<Vec<_>>());
    let lf_interp = interp(&theta, &theta_fine, &lf_fine);

    // Compute initial conditions
    let mej0: Vec<f64> = energy_interp.iter().zip(lf_interp.iter())
        .map(|(e, lf)| e / (lf - 1.0))
        .collect();
    let beta0: Vec<f64> = lf_interp.iter()
        .map(|lf| (1.0 - 1.0 / (lf * lf)).sqrt())
        .collect();
    let r0: Vec<f64> = beta0.iter()
        .map(|b| b * C_SPEED * tmin)
        .collect();
    let msw0: Vec<f64> = r0.iter()
        .map(|r| nwind * MASS_P * r / 1e17 * 1e51 + n0 * MASS_P * r * r * r / 3.0)
        .collect();
    let eb0: Vec<f64> = energy_interp.iter().zip(mej0.iter()).zip(msw0.iter())
        .map(|((e, m), ms)| e + m + ms)
        .collect();
    let ht0 = vec![0.0; ncells];

    JetConfig {
        theta_edge,
        eb: eb0,
        ht: ht0,
        msw: msw0,
        mej: mej0,
        r: r0,
        nwind,
        nism: n0,
        tmin,
        tmax,
        rtol: 1e-6,
        cfl: 0.9,
        spread: true,
        cal_level: 1,
    }
}

/// Linear interpolation (like numpy.interp).
fn interp(x_new: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
    x_new.iter().map(|&xn| {
        if xn <= x[0] {
            return y[0];
        }
        if xn >= x[x.len() - 1] {
            return y[y.len() - 1];
        }
        let mut lo = 0;
        let mut hi = x.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if xn > x[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let frac = (xn - x[lo]) / (x[hi] - x[lo]);
        y[lo] + frac * (y[hi] - y[lo])
    }).collect()
}

// ─── Tests ───

#[test]
fn test_pde_solver_produces_valid_output() {
    let config = tophat_config();
    let ncells = config.eb.len();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();

    // Must have at least 2 time steps (initial + evolution)
    assert!(sim.ts.len() >= 2, "PDE should produce multiple time steps");

    // Time should be monotonically increasing
    for i in 1..sim.ts.len() {
        assert!(sim.ts[i] > sim.ts[i - 1], "Time must be monotonically increasing");
    }

    // Should have 5 variables (msw, mej, beta_gamma_sq, beta_th, R)
    assert_eq!(sim.ys.len(), 5);
    for var in &sim.ys {
        assert_eq!(var.len(), ncells);
        for cell in var {
            assert_eq!(cell.len(), sim.ts.len());
        }
    }

    // All values should be finite
    for var in &sim.ys {
        for cell in var {
            for &val in cell {
                assert!(val.is_finite(), "PDE solution contains non-finite value");
            }
        }
    }
}

#[test]
fn test_interpolation_on_pde_solution() {
    let config = tophat_config();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();

    let theta = sim.get_theta().clone();
    let tool = Tool::new(config.nwind, config.nism, config.rtol, config.cal_level);
    let interpolator = Interpolator::new(&theta, &sim.ts);

    // Interpolate at the midpoint of the time range, at the on-axis cell
    let t_mid = sim.ts[sim.ts.len() / 2];
    let theta_on = theta[0];

    // beta_gamma_sq (index 2) should be positive for the core
    let bg_sq = interpolator.interpolate_y(t_mid, theta_on, 2, &sim.ys, &sim.ts, &theta, &tool);
    assert!(bg_sq > 0.0, "beta_gamma_sq should be positive on-axis, got {}", bg_sq);
    assert!(bg_sq.is_finite());

    // R (index 4) should be positive and growing
    let r_early = interpolator.interpolate_y(sim.ts[1], theta_on, 4, &sim.ys, &sim.ts, &theta, &tool);
    let r_late = interpolator.interpolate_y(t_mid, theta_on, 4, &sim.ys, &sim.ts, &theta, &tool);
    assert!(r_late > r_early, "Radius should grow over time");

    // Msw (index 0) should be non-negative
    let msw = interpolator.interpolate_y(t_mid, theta_on, 0, &sim.ys, &sim.ts, &theta, &tool);
    assert!(msw >= 0.0, "Swept-up mass should be non-negative");
}

#[test]
fn test_luminosity_positive_and_finite() {
    let config = tophat_config();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();

    let theta = sim.get_theta().clone();
    let tool = Tool::new(config.nwind, config.nism, config.rtol, config.cal_level);
    let eats = EATS::new(&theta, &sim.ts);
    let mut afterglow = Afterglow::new();

    // Configure with quick-start parameters
    let mut param = HashMap::new();
    param.insert("eps_e".into(), 0.1);
    param.insert("eps_b".into(), 0.01);
    param.insert("p".into(), 2.17);
    param.insert("theta_v".into(), 0.0);
    param.insert("d".into(), 474.33);
    param.insert("z".into(), 0.1);
    afterglow.config_parameters(param);
    afterglow.config_intensity("sync");

    // Compute luminosity at 1 day, 1 keV (X-ray)
    let tobs = 86400.0; // 1 day in seconds
    let nu = 1e18;      // ~1 keV
    let luminosity = afterglow.luminosity(
        tobs, nu, 1e-3, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );

    assert!(luminosity > 0.0, "Luminosity should be positive, got {}", luminosity);
    assert!(luminosity.is_finite(), "Luminosity should be finite");

    // Sanity: spectral luminosity should be in a physically plausible range
    assert!(luminosity > 1e20, "Luminosity suspiciously low: {:.3e}", luminosity);
    assert!(luminosity < 1e55, "Luminosity suspiciously high: {:.3e}", luminosity);
}

#[test]
fn test_flux_density_decreases_at_late_times() {
    let config = tophat_config();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();

    let theta = sim.get_theta().clone();
    let tool = Tool::new(config.nwind, config.nism, config.rtol, config.cal_level);
    let eats = EATS::new(&theta, &sim.ts);
    let mut afterglow = Afterglow::new();

    let mut param = HashMap::new();
    param.insert("eps_e".into(), 0.1);
    param.insert("eps_b".into(), 0.01);
    param.insert("p".into(), 2.17);
    param.insert("theta_v".into(), 0.0);
    param.insert("d".into(), 474.33);
    param.insert("z".into(), 0.1);
    afterglow.config_parameters(param);
    afterglow.config_intensity("sync");

    let nu = 1e18;

    // Compare luminosity at 1 day vs 100 days (on-axis top-hat should be decaying)
    let l_early = afterglow.luminosity(
        86400.0, nu, 1e-2, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );
    let l_late = afterglow.luminosity(
        86400.0 * 100.0, nu, 1e-2, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );

    assert!(
        l_early > l_late,
        "On-axis top-hat X-ray flux should decay: L(1d) = {:.3e}, L(100d) = {:.3e}",
        l_early, l_late,
    );
}
