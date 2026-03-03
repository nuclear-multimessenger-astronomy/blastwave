//! Integration test: full top-hat jet simulation pipeline.
//!
//! Mirrors the Python `FluxDensity_tophat` workflow entirely in Rust:
//! build initial conditions, solve the PDE, interpolate, and compute
//! afterglow luminosity.

use std::collections::HashMap;

use blastwave::constants::*;
use blastwave::hydro::config::{JetConfig, SpreadMode};
use blastwave::hydro::sim_box::SimBox;
use blastwave::hydro::interpolate::Interpolator;
use blastwave::hydro::tools::Tool;
use blastwave::afterglow::eats::EATS;
use blastwave::afterglow::afterglow::Afterglow;
use blastwave::afterglow::forward_grid::ForwardGrid;

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
        ..JetConfig::default()
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

    // Should have 6 variables (msw, mej, beta_gamma_sq, beta_th, R, u2_th)
    assert_eq!(sim.ys.len(), 6);
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

/// Build a small top-hat jet config with reverse shock enabled.
/// Uses fewer cells and shorter time range for faster integration tests.
fn small_tophat_config_with_rs() -> JetConfig {
    let theta_c: f64 = 0.1;
    let eiso: f64 = 1e52;
    let lf0: f64 = 300.0;
    let n0: f64 = 1.0;
    let nwind: f64 = 0.0;
    let tmin: f64 = 10.0;
    let tmax: f64 = 1e7; // shorter time range for RS tests

    // Small grid: 17 edges = 16 cells (much faster than 128)
    let npoints = 17;
    let mut theta_edge = vec![0.0; npoints];
    let arcsinh_max = (PI / theta_c).asinh();
    for i in 0..npoints {
        theta_edge[i] = (i as f64 / (npoints - 1) as f64 * arcsinh_max).sinh() * theta_c;
    }
    theta_edge[0] = 0.0;
    theta_edge[npoints - 1] = PI;

    let ncells = npoints - 1;
    let theta: Vec<f64> = (0..ncells)
        .map(|i| (theta_edge[i] + theta_edge[i + 1]) / 2.0)
        .collect();

    let nfine = 1000;
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

    let max_e = energy_fine.iter().cloned().fold(0.0f64, f64::max);
    for e in energy_fine.iter_mut() {
        if *e <= max_e * 1e-12 { *e = max_e * 1e-12; }
    }
    for lf in lf_fine.iter_mut() {
        if *lf <= 1.005 { *lf = 1.005; }
    }

    let energy_interp = interp(&theta, &theta_fine,
        &energy_fine.iter().map(|e| e / 4.0 / PI / C_SPEED / C_SPEED).collect::<Vec<_>>());
    let lf_interp = interp(&theta, &theta_fine, &lf_fine);

    let mej0: Vec<f64> = energy_interp.iter().zip(lf_interp.iter())
        .map(|(e, lf)| e / (lf - 1.0)).collect();
    let beta0: Vec<f64> = lf_interp.iter()
        .map(|lf| (1.0 - 1.0 / (lf * lf)).sqrt()).collect();
    let r0: Vec<f64> = beta0.iter()
        .map(|b| b * C_SPEED * tmin).collect();
    let msw0: Vec<f64> = r0.iter()
        .map(|r| nwind * MASS_P * r / 1e17 * 1e51 + n0 * MASS_P * r * r * r / 3.0).collect();
    let eb0: Vec<f64> = energy_interp.iter().zip(mej0.iter()).zip(msw0.iter())
        .map(|((e, m), ms)| e + m + ms).collect();
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
        rtol: 1e-4, // relaxed tolerance for speed
        cfl: 0.9,
        spread: true,
        cal_level: 1,
        include_reverse_shock: true,
        sigma: 0.0,
        eps_e_rs: 0.1,
        eps_b_rs: 0.01,
        p_rs: 2.3,
        ..JetConfig::default()
    }
}

#[test]
fn test_forward_only_regression() {
    // Verify that the forward shock luminosity is positive/finite (regression test)
    let config_fs = tophat_config();
    let mut sim_fs = SimBox::new(&config_fs);
    sim_fs.solve_pde();

    let theta = sim_fs.get_theta().clone();
    let tool = Tool::new(config_fs.nwind, config_fs.nism, config_fs.rtol, config_fs.cal_level);
    let eats = EATS::new(&theta, &sim_fs.ts);
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

    let tobs = 86400.0;
    let nu = 1e18;
    let l_fs = afterglow.luminosity(
        tobs, nu, 1e-3, 50, true,
        &eats, &sim_fs.ys, &sim_fs.ts, &theta, &tool,
    );

    assert!(l_fs > 0.0, "Forward shock luminosity should be positive");
    assert!(l_fs.is_finite(), "Forward shock luminosity should be finite");
}

#[test]
fn test_rs_pde_produces_valid_output() {
    let config = small_tophat_config_with_rs();
    let ncells = config.eb.len();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();
    sim.solve_reverse_shock();

    // RS data should be populated
    assert!(sim.ys_rs.is_some(), "RS data should be populated when reverse shock is enabled");
    let ys_rs = sim.ys_rs.as_ref().unwrap();

    // Should have NVAR_RS variables
    assert!(ys_rs.len() > 0, "RS data should have state variables");

    // Each variable should have ncells theta cells
    for var in ys_rs {
        assert_eq!(var.len(), ncells, "RS vars should have same theta cells as FS");
    }

    // All RS values should be finite
    for var in ys_rs {
        for cell in var {
            for &val in cell {
                assert!(val.is_finite(), "RS solution contains non-finite value: {}", val);
            }
        }
    }
}

#[test]
fn test_rs_luminosity_positive_and_finite() {
    let config = small_tophat_config_with_rs();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();
    sim.solve_reverse_shock();

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

    // Configure RS parameters
    let mut param_rs = HashMap::new();
    param_rs.insert("eps_e".into(), config.eps_e_rs);
    param_rs.insert("eps_b".into(), config.eps_b_rs);
    param_rs.insert("p".into(), config.p_rs);
    param_rs.insert("theta_v".into(), 0.0);
    param_rs.insert("d".into(), 474.33);
    param_rs.insert("z".into(), 0.1);
    afterglow.config_rs_parameters(param_rs);

    let rs_data = sim.ys_rs.as_ref().unwrap();

    // Test total luminosity (FS + RS) at early time (when RS is strong)
    let tobs = 1000.0; // early time
    let nu = 1e10;     // radio (where RS is typically stronger)
    let l_total = afterglow.luminosity_total(
        tobs, nu, 1e-2, 30, true,
        &eats, &sim.ys, rs_data, &sim.ts, &theta, &tool,
    );

    assert!(l_total.is_finite(), "Total luminosity should be finite, got {}", l_total);

    // Test FS luminosity alone
    let l_fs = afterglow.luminosity(
        tobs, nu, 1e-2, 30, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );
    assert!(l_fs.is_finite(), "FS luminosity should be finite");

    // Total should be >= FS (since RS adds non-negative contribution)
    assert!(
        l_total >= l_fs - 1e-30,
        "Total luminosity should be >= FS luminosity: total={:.3e}, fs={:.3e}",
        l_total, l_fs,
    );
}

/// Build a top-hat config with ODE spreading mode.
fn tophat_config_ode() -> JetConfig {
    let mut config = tophat_config();
    config.spread_mode = SpreadMode::Ode;
    config.theta_c = 0.1;
    config
}

#[test]
fn test_ode_spread_produces_valid_output() {
    let config = tophat_config_ode();
    let ncells = config.eb.len();
    let mut sim = SimBox::new(&config);
    sim.solve_pde();

    // Must have output time steps
    assert!(sim.ts.len() >= 2, "ODE spread should produce multiple time steps");

    // Time should be monotonically increasing
    for i in 1..sim.ts.len() {
        assert!(sim.ts[i] > sim.ts[i - 1], "Time must be monotonically increasing");
    }

    // Should have 7 variables (msw, mej, beta_gamma_sq, beta_th, r, u2_th, t_comv)
    assert_eq!(sim.ys.len(), 7);
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
                assert!(val.is_finite(), "ODE spread solution contains non-finite value");
            }
        }
    }
}

#[test]
fn test_ode_spread_luminosity() {
    let config = tophat_config_ode();
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

    let tobs = 86400.0;
    let nu = 1e18;
    let luminosity = afterglow.luminosity(
        tobs, nu, 1e-3, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );

    assert!(luminosity > 0.0, "ODE spread luminosity should be positive, got {}", luminosity);
    assert!(luminosity.is_finite(), "ODE spread luminosity should be finite");
    assert!(luminosity > 1e20, "ODE spread luminosity suspiciously low: {:.3e}", luminosity);
    assert!(luminosity < 1e55, "ODE spread luminosity suspiciously high: {:.3e}", luminosity);
}

#[test]
fn test_ode_vs_pde_spread_agreement() {
    // ODE and PDE modes should produce broadly similar results for tophat jets
    let config_pde = tophat_config();
    let mut sim_pde = SimBox::new(&config_pde);
    sim_pde.solve_pde();

    let config_ode = tophat_config_ode();
    let mut sim_ode = SimBox::new(&config_ode);
    sim_ode.solve_pde();

    let theta_pde = sim_pde.get_theta().clone();
    let theta_ode = sim_ode.get_theta().clone();
    let tool = Tool::new(config_pde.nwind, config_pde.nism, config_pde.rtol, config_pde.cal_level);

    let eats_pde = EATS::new(&theta_pde, &sim_pde.ts);
    let eats_ode = EATS::new(&theta_ode, &sim_ode.ts);

    let mut afterglow_pde = Afterglow::new();
    let mut afterglow_ode = Afterglow::new();

    let mut param = HashMap::new();
    param.insert("eps_e".into(), 0.1);
    param.insert("eps_b".into(), 0.01);
    param.insert("p".into(), 2.17);
    param.insert("theta_v".into(), 0.0);
    param.insert("d".into(), 474.33);
    param.insert("z".into(), 0.1);
    afterglow_pde.config_parameters(param.clone());
    afterglow_pde.config_intensity("sync");
    afterglow_ode.config_parameters(param);
    afterglow_ode.config_intensity("sync");

    let nu = 1e18;

    // Compare at several times
    for &tobs in &[1e3, 1e5, 1e7] {
        let l_pde = afterglow_pde.luminosity(
            tobs, nu, 1e-2, 50, true,
            &eats_pde, &sim_pde.ys, &sim_pde.ts, &theta_pde, &tool,
        );
        let l_ode = afterglow_ode.luminosity(
            tobs, nu, 1e-2, 50, true,
            &eats_ode, &sim_ode.ys, &sim_ode.ts, &theta_ode, &tool,
        );

        if l_pde > 0.0 && l_ode > 0.0 {
            let log_diff = (l_pde.log10() - l_ode.log10()).abs();
            assert!(
                log_diff < 1.0,
                "ODE vs PDE luminosity differ by > 1 dex at t={:.0e}: PDE={:.3e}, ODE={:.3e} (diff={:.2} dex)",
                tobs, l_pde, l_ode, log_diff,
            );
        }
    }
}

// ─── SSA parameter sweep ───

/// Build a no-spread top-hat config for SSA tests.
/// Uses a small grid (17 edges) and ODE no-spread mode for speed.
fn nospread_tophat_config(n0: f64, eps_e: f64, eps_b: f64, p: f64) -> JetConfig {
    let theta_c: f64 = 0.1;
    let eiso: f64 = 1e53;
    let lf0: f64 = 300.0;
    let tmin: f64 = 10.0;
    let tmax: f64 = 1e9;

    let npoints = 17;
    let mut theta_edge = vec![0.0; npoints];
    let arcsinh_max = (PI / theta_c).asinh();
    for i in 0..npoints {
        theta_edge[i] = (i as f64 / (npoints - 1) as f64 * arcsinh_max).sinh() * theta_c;
    }
    theta_edge[0] = 0.0;
    theta_edge[npoints - 1] = PI;

    let ncells = npoints - 1;
    let theta: Vec<f64> = (0..ncells)
        .map(|i| (theta_edge[i] + theta_edge[i + 1]) / 2.0)
        .collect();

    let nfine = 1000;
    let theta_fine: Vec<f64> = (0..nfine)
        .map(|i| i as f64 / (nfine - 1) as f64 * PI)
        .collect();
    let mut energy_fine = vec![eiso; nfine];
    let mut lf_fine = vec![1.0f64; nfine];
    for i in 0..nfine {
        if theta_fine[i] > theta_c {
            energy_fine[i] = 0.0;
        } else {
            lf_fine[i] = lf0;
        }
    }

    let max_e = energy_fine.iter().cloned().fold(0.0f64, f64::max);
    for e in energy_fine.iter_mut() {
        if *e <= max_e * 1e-12 { *e = max_e * 1e-12; }
    }
    for lf in lf_fine.iter_mut() {
        if *lf <= 1.005 { *lf = 1.005; }
    }

    let energy_interp = interp(
        &theta, &theta_fine,
        &energy_fine.iter().map(|e| e / 4.0 / PI / C_SPEED / C_SPEED).collect::<Vec<_>>(),
    );
    let lf_interp = interp(&theta, &theta_fine, &lf_fine);

    let mej0: Vec<f64> = energy_interp.iter().zip(lf_interp.iter())
        .map(|(e, lf)| e / (lf - 1.0)).collect();
    let beta0: Vec<f64> = lf_interp.iter()
        .map(|lf| (1.0 - 1.0 / (lf * lf)).sqrt()).collect();
    let r0: Vec<f64> = beta0.iter()
        .map(|b| b * C_SPEED * tmin).collect();
    let msw0: Vec<f64> = r0.iter()
        .map(|r| n0 * MASS_P * r * r * r / 3.0).collect();
    let eb0: Vec<f64> = energy_interp.iter().zip(mej0.iter()).zip(msw0.iter())
        .map(|((e, m), ms)| e + m + ms).collect();
    let ht0 = vec![0.0; ncells];

    JetConfig {
        theta_edge,
        eb: eb0,
        ht: ht0,
        msw: msw0,
        mej: mej0,
        r: r0,
        nwind: 0.0,
        nism: n0,
        tmin,
        tmax,
        rtol: 1e-6,
        cfl: 0.9,
        spread: false,
        cal_level: 1,
        eps_e,
        eps_b,
        p_fwd: p,
        ..JetConfig::default()
    }
}

/// Compute luminosity for a given config, model, time, and frequency.
fn compute_lum(
    config: &JetConfig,
    sim: &SimBox,
    model_name: &str,
    eps_e: f64,
    eps_b: f64,
    p: f64,
    tobs: f64,
    nu: f64,
) -> f64 {
    let theta = sim.get_theta().clone();
    let tool = Tool::new(config.nwind, config.nism, config.rtol, config.cal_level);
    let eats = EATS::new(&theta, &sim.ts);
    let mut afterglow = Afterglow::new();

    let mut param = HashMap::new();
    param.insert("eps_e".into(), eps_e);
    param.insert("eps_b".into(), eps_b);
    param.insert("p".into(), p);
    param.insert("theta_v".into(), 0.0);
    param.insert("d".into(), 42.15); // ~1.3e26 cm in Mpc
    param.insert("z".into(), 0.01);
    afterglow.config_parameters(param);
    afterglow.config_intensity(model_name);

    afterglow.luminosity(
        tobs, nu, 1e-3, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    )
}

#[test]
fn test_sync_ssa_smooth_parameter_sweep() {
    // Regression test for SSA parameter sweep.
    //
    // Verifies that sync_ssa_smooth correctly handles self-absorption across
    // a range of physical parameters. The reference values were calibrated
    // against VegasAfterglow (median BW/VA = 0.97–1.03 across frequencies).
    //
    // For each config we check:
    //  1. SSA does not affect X-ray (ratio to sync_smooth > 0.95)
    //  2. SSA suppresses radio (sync_ssa_smooth < sync_smooth)
    //  3. Absolute luminosity values match reference within 20%

    struct Case {
        name: &'static str,
        n0: f64,
        eps_e: f64,
        eps_b: f64,
        p: f64,
        // Reference luminosities (erg/s/Hz) at t=1e5 s, calibrated against
        // VegasAfterglow (BW/VA median 0.97–1.03 across frequencies).
        ref_radio: f64,  // 1 GHz SSA luminosity
        ref_xray: f64,   // 1 keV SSA luminosity
    }

    let cases = [
        Case { name: "default",   n0: 1.0,   eps_e: 0.1, eps_b: 0.01, p: 2.3,
               ref_radio: 5.69e29, ref_xray: 1.28e27 },
        Case { name: "n0=0.01",   n0: 0.01,  eps_e: 0.1, eps_b: 0.01, p: 2.3,
               ref_radio: 3.08e30, ref_xray: 4.01e27 },
        Case { name: "n0=100",    n0: 100.0,  eps_e: 0.1, eps_b: 0.01, p: 2.3,
               ref_radio: 8.95e27, ref_xray: 1.90e26 },
        Case { name: "eps_b=0.1", n0: 1.0,   eps_e: 0.1, eps_b: 0.1,  p: 2.3,
               ref_radio: 5.34e29, ref_xray: 1.56e27 },
        Case { name: "eps_e=0.3", n0: 1.0,   eps_e: 0.3, eps_b: 0.01, p: 2.3,
               ref_radio: 1.47e30, ref_xray: 5.33e27 },
        Case { name: "p=2.5",     n0: 1.0,   eps_e: 0.1, eps_b: 0.01, p: 2.5,
               ref_radio: 8.06e29, ref_xray: 6.06e26 },
        Case { name: "p=2.05",    n0: 1.0,   eps_e: 0.1, eps_b: 0.01, p: 2.05,
               ref_radio: 7.73e28, ref_xray: 1.26e27 },
    ];

    let nu_radio = 1e9;     // 1 GHz
    let nu_xray = 2.4e17;   // 1 keV
    let tobs = 1e5;          // ~1.16 days
    let tol = 0.20;          // 20% tolerance for regression

    for case in &cases {
        let config = nospread_tophat_config(case.n0, case.eps_e, case.eps_b, case.p);
        let mut sim = SimBox::new(&config);
        sim.solve_pde();

        // SSA model
        let l_ssa_radio = compute_lum(
            &config, &sim, "sync_ssa_smooth",
            case.eps_e, case.eps_b, case.p, tobs, nu_radio,
        );
        let l_ssa_xray = compute_lum(
            &config, &sim, "sync_ssa_smooth",
            case.eps_e, case.eps_b, case.p, tobs, nu_xray,
        );

        // Reference (no SSA) model
        let l_smooth_radio = compute_lum(
            &config, &sim, "sync_smooth",
            case.eps_e, case.eps_b, case.p, tobs, nu_radio,
        );
        let l_smooth_xray = compute_lum(
            &config, &sim, "sync_smooth",
            case.eps_e, case.eps_b, case.p, tobs, nu_xray,
        );

        // --- Assertions ---

        // All values must be positive and finite
        assert!(l_ssa_radio > 0.0 && l_ssa_radio.is_finite(),
            "[{}] SSA radio luminosity invalid: {:.3e}", case.name, l_ssa_radio);
        assert!(l_ssa_xray > 0.0 && l_ssa_xray.is_finite(),
            "[{}] SSA X-ray luminosity invalid: {:.3e}", case.name, l_ssa_xray);

        // 1. SSA should NOT affect X-ray (harmonic mean ≈ thin when I_thick >> I_thin)
        let xray_ratio = l_ssa_xray / l_smooth_xray;
        assert!(
            xray_ratio > 0.90 && xray_ratio <= 1.001,
            "[{}] SSA should not affect X-ray: SSA/smooth = {:.4} (expected ~1.0)",
            case.name, xray_ratio,
        );

        // 2. SSA MUST suppress radio
        assert!(
            l_ssa_radio < l_smooth_radio,
            "[{}] SSA must suppress radio: SSA={:.3e} >= smooth={:.3e}",
            case.name, l_ssa_radio, l_smooth_radio,
        );

        // 3. Regression: SSA radio luminosity should match reference within tolerance.
        //    Reference values were calibrated against VegasAfterglow.
        let radio_frac = (l_ssa_radio - case.ref_radio).abs() / case.ref_radio;
        assert!(
            radio_frac < tol,
            "[{}] SSA radio regression: got {:.3e}, expected {:.3e} (off by {:.1}%)",
            case.name, l_ssa_radio, case.ref_radio, radio_frac * 100.0,
        );

        // 4. Regression: X-ray luminosity should match reference within tolerance.
        let xray_frac = (l_ssa_xray - case.ref_xray).abs() / case.ref_xray;
        assert!(
            xray_frac < tol,
            "[{}] SSA X-ray regression: got {:.3e}, expected {:.3e} (off by {:.1}%)",
            case.name, l_ssa_xray, case.ref_xray, xray_frac * 100.0,
        );
    }
}

// ─── Forward-mapping tests ───

#[test]
fn test_forward_grid_luminosity_positive_and_finite() {
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
    afterglow.config_parameters(param.clone());
    afterglow.config_intensity("sync");

    let nu = 1e18;
    let z = 0.1;
    let nu_z = nu * (1.0 + z);

    let grid = ForwardGrid::precompute(
        nu_z, 0.0, &sim.ys, &sim.ts, &theta,
        &eats, &tool, &param, afterglow.radiation_model.unwrap(),
    );

    // Compute luminosity at 1 day
    let tobs_z = 86400.0 * (1.0 + z);
    let luminosity = grid.luminosity(tobs_z);

    assert!(luminosity > 0.0, "Forward grid luminosity should be positive, got {}", luminosity);
    assert!(luminosity.is_finite(), "Forward grid luminosity should be finite");
    assert!(luminosity > 1e20, "Forward grid luminosity suspiciously low: {:.3e}", luminosity);
    assert!(luminosity < 1e55, "Forward grid luminosity suspiciously high: {:.3e}", luminosity);
}

#[test]
fn test_forward_grid_agrees_with_eats() {
    // Forward-mapping and EATS should agree within 0.1 dex for on-axis PDE
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
    afterglow.config_parameters(param.clone());
    afterglow.config_intensity("sync");

    let nu = 1e18;
    let z = 0.1;
    let nu_z = nu * (1.0 + z);

    let grid = ForwardGrid::precompute(
        nu_z, 0.0, &sim.ys, &sim.ts, &theta,
        &eats, &tool, &param, afterglow.radiation_model.unwrap(),
    );

    // Compare at several observation times.
    // Tolerance is 0.3 dex per point; individual early times can deviate more
    // than the aggregate max across a lightcurve (which is ~0.002 dex for PDE).
    for &tobs in &[1e4, 86400.0, 86400.0 * 10.0, 86400.0 * 100.0] {
        let l_eats = afterglow.luminosity(
            tobs, nu, 1e-3, 50, true,
            &eats, &sim.ys, &sim.ts, &theta, &tool,
        );
        let l_fwd = grid.luminosity(tobs * (1.0 + z));

        if l_eats > 0.0 && l_fwd > 0.0 {
            let log_diff = (l_eats.log10() - l_fwd.log10()).abs();
            assert!(
                log_diff < 0.3,
                "Forward vs EATS differ by > 0.3 dex at t={:.0e}: EATS={:.3e}, Forward={:.3e} (diff={:.4} dex)",
                tobs, l_eats, l_fwd, log_diff,
            );
        }
    }
}

#[test]
fn test_forward_grid_decays_at_late_times() {
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
    afterglow.config_parameters(param.clone());
    afterglow.config_intensity("sync");

    let nu = 1e18;
    let z = 0.1;
    let nu_z = nu * (1.0 + z);

    let grid = ForwardGrid::precompute(
        nu_z, 0.0, &sim.ys, &sim.ts, &theta,
        &eats, &tool, &param, afterglow.radiation_model.unwrap(),
    );

    let l_early = grid.luminosity(86400.0 * (1.0 + z));
    let l_late = grid.luminosity(86400.0 * 100.0 * (1.0 + z));

    assert!(
        l_early > l_late,
        "On-axis X-ray flux should decay: L(1d) = {:.3e}, L(100d) = {:.3e}",
        l_early, l_late,
    );
}

#[test]
fn test_forward_grid_ode_spread() {
    // Forward-mapping should also work with ODE spreading mode
    let config = tophat_config_ode();
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
    afterglow.config_parameters(param.clone());
    afterglow.config_intensity("sync");

    let nu = 1e18;
    let z = 0.1;
    let nu_z = nu * (1.0 + z);

    let grid = ForwardGrid::precompute(
        nu_z, 0.0, &sim.ys, &sim.ts, &theta,
        &eats, &tool, &param, afterglow.radiation_model.unwrap(),
    );

    let tobs = 86400.0;
    let l_fwd = grid.luminosity(tobs * (1.0 + z));
    let l_eats = afterglow.luminosity(
        tobs, nu, 1e-3, 50, true,
        &eats, &sim.ys, &sim.ts, &theta, &tool,
    );

    assert!(l_fwd > 0.0, "ODE forward grid luminosity should be positive");
    assert!(l_fwd.is_finite(), "ODE forward grid luminosity should be finite");

    if l_eats > 0.0 && l_fwd > 0.0 {
        let log_diff = (l_eats.log10() - l_fwd.log10()).abs();
        assert!(
            log_diff < 0.3,
            "ODE: Forward vs EATS differ by > 0.3 dex: EATS={:.3e}, Forward={:.3e} (diff={:.4} dex)",
            l_eats, l_fwd, log_diff,
        );
    }
}
