//! Internal shock emission from trailing shell collisions.
//!
//! When a trailing shell catches the decelerating blast wave, the collision
//! produces a forward-reverse shock pair. The forward shock energizes the
//! blast wave (handled by the hydro solver's energy injection). This module
//! computes the **reverse shock emission** from the collision-heated shell
//! material, which produces a bright transient "flash".
//!
//! Physics model:
//! - At collision, a reverse shock propagates through the trailing shell
//! - The shocked shell material has:
//!   - Thermal energy: e_th = (Γ_rel - 1) × m_shell × c²
//!   - B-field: B' = √(8π ε_B e'_int)
//!   - γ_m = ε_e (p-2)/(p-1) (m_p/m_e) (Γ_rel - 1)
//! - After collision, the material coasts with the merged blast wave
//!   and cools via adiabatic expansion:
//!   - n' ∝ R⁻³, e' ∝ R⁻⁴, B' ∝ R⁻², γ_m ∝ R⁻¹
//!
//! Reference: Kobayashi et al. (1997), Daigne & Mochkovitch (1998),
//!            Sari & Meszaros (2000)

use crate::constants::*;
use crate::hydro::config::CollisionEvent;
use crate::hydro::tools::Tool;
use crate::afterglow::blast::{Blast, ShockType};
use crate::afterglow::eats::EATS;
use crate::afterglow::models::{Dict, RadiationModel};
use crate::afterglow::forward_grid::{ForwardGrid, compute_dcos_theta, build_phi_grid, compute_bg_threshold};
use rayon::prelude::*;

/// Build a ForwardGrid for collision emission.
///
/// For each collision event, builds a synthetic emission component that
/// decays with adiabatic expansion. Uses the same EATS geometry as
/// the forward shock (the collision material co-moves with the blast).
///
/// The collision emission is computed by creating a modified Blast state
/// at each (cell, time) point with the collision shock parameters, then
/// evaluating the radiation model.
pub fn precompute_collision_grid(
    nu_z: f64,
    theta_v: f64,
    y_data: &[Vec<Vec<f64>>],
    t_data: &[f64],
    theta_data: &[f64],
    eats: &EATS,
    tool: &Tool,
    param_coll: &Dict,
    radiation_model: RadiationModel,
    collision_events: &[CollisionEvent],
) -> ForwardGrid {
    if collision_events.is_empty() {
        return ForwardGrid::empty();
    }

    let ntheta = theta_data.len();
    let nt = t_data.len();
    let nvar = y_data.len();

    let all_dcos = compute_dcos_theta(theta_data);
    let bg_threshold = compute_bg_threshold(y_data, ntheta);
    let (phis, dphis) = build_phi_grid(theta_v, theta_data, y_data);

    // Build a map: cell_idx -> Vec<&CollisionEvent>
    // Each theta cell may have 0, 1, or more collision events
    let mut cell_events: Vec<Vec<&CollisionEvent>> = vec![Vec::new(); ntheta];
    for ev in collision_events {
        if ev.cell_idx < ntheta {
            cell_events[ev.cell_idx].push(ev);
        }
    }

    // Build cell work items: only cells that have collision events
    let mut cell_specs: Vec<(usize, usize)> = Vec::new();
    for j in 0..ntheta {
        if cell_events[j].is_empty() { continue; }
        if y_data[2][j].iter().copied().fold(0.0f64, f64::max) < bg_threshold {
            continue;
        }
        for phi_idx in 0..phis.len() {
            cell_specs.push((j, phi_idx));
        }
    }

    // Get microphysics from param dict
    let eps_e_coll = *param_coll.get("eps_e").unwrap_or(&0.1);
    let eps_b_coll = *param_coll.get("eps_b").unwrap_or(&0.01);
    let p_coll = *param_coll.get("p").unwrap_or(&2.3);

    let cells: Vec<_> = cell_specs.par_iter()
        .map(|&(j, phi_idx)| {
            let theta = theta_data[j];
            let phi = phis[phi_idx];
            let mu = theta.cos() * theta_v.cos() + theta.sin() * phi.cos() * theta_v.sin();
            let events = &cell_events[j];

            let mut cell_tobs = Vec::with_capacity(nt);
            let mut cell_dl = Vec::with_capacity(nt);

            for k in 0..nt {
                let t_lab = t_data[k];
                let r = y_data[4][j][k];
                if r <= 0.0 { continue; }

                let bg_sq = y_data[2][j][k].max(0.0);
                if bg_sq < bg_threshold { continue; }
                let gamma = (1.0 + bg_sq).sqrt();

                // Sum emission from all collision events active at this time
                let mut total_intensity = 0.0;

                for ev in events.iter() {
                    if t_lab <= ev.t_lab { continue; }

                    // Expansion factor since collision
                    let r_ratio = r / ev.r_coll;
                    if r_ratio <= 0.0 || !r_ratio.is_finite() { continue; }

                    // Adiabatic evolution of collision-heated material
                    // The material co-moves with the blast (same R, same Γ)
                    // but its internal energy decays as V'^{-4/3} ∝ R^{-4}
                    let f_r = ev.r_coll / r;  // < 1 for expansion
                    let f_r2 = f_r * f_r;
                    let f_r3 = f_r2 * f_r;
                    let f_r4 = f_r3 * f_r;

                    // Post-shock conditions at collision time
                    let gamma_rel = ev.gamma_rel;
                    if gamma_rel <= 1.0 { continue; }

                    // Internal energy density at collision
                    // e'_0 = (Γ_rel - 1) × n'_comv × m_p × c²
                    let e_density_0 = (gamma_rel - 1.0) * ev.n_sh_comv * MASS_P * C_SPEED * C_SPEED;

                    // Evolved quantities
                    let n_comv = ev.n_sh_comv * f_r3;     // n' ∝ R⁻³
                    let e_density = e_density_0 * f_r4;   // e' ∝ R⁻⁴
                    let gamma_th = e_density / (n_comv * MASS_P * C_SPEED * C_SPEED) + 1.0;

                    if gamma_th <= 1.0 || e_density <= 0.0 || n_comv <= 0.0 { continue; }

                    // Shell thickness: comoving thickness grows as R/R_0
                    // Initial thickness: Δ'_0 ≈ m_shell / (n_comv_0 × m_p × R²)
                    let dr_0 = ev.m_sh_per_sr / (ev.n_sh_comv * MASS_P * ev.r_coll * ev.r_coll);
                    let dr = dr_0 * r_ratio;  // Δ' grows linearly with R

                    // Build a Blast state for the collision-heated material
                    // Use the same bulk motion as the forward shock (co-moving)
                    let val = [
                        y_data[0][j][k],
                        y_data[1][j][k],
                        y_data[2][j][k],
                        y_data[3][j][k],
                        y_data[4][j][k],
                        y_data[5][j][k],
                        if nvar > 6 { y_data[6][j][k] } else { 0.0 },
                        t_lab,
                    ];

                    let mut blast = Blast::default();
                    eats.derive_blast(theta, phi, theta_v, &val, tool, &mut blast);

                    // Override thermodynamic quantities with collision shock values
                    blast.shock_type = ShockType::Forward;  // use FS radiation model
                    blast.gamma_th = gamma_th;
                    blast.e_density = e_density;
                    blast.n_blast = n_comv;
                    blast.dr = dr;

                    let nu_src = nu_z / blast.doppler;
                    let intensity = radiation_model(nu_src, param_coll, &blast);
                    total_intensity += intensity;
                }

                if total_intensity > 0.0 {
                    // Derive Doppler factor from blast state
                    let beta = bg_sq.sqrt() / gamma;
                    let doppler = 1.0 / (gamma * (1.0 - beta * mu));

                    cell_tobs.push(t_lab - r * mu / C_SPEED);
                    cell_dl.push(total_intensity * r * r * doppler * doppler * doppler);
                }
            }

            if cell_tobs.is_empty() {
                return (Vec::new(), Vec::new(), 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
            }

            let lg2_t = to_lg2(&cell_tobs);
            let lg2_dl = to_lg2(&cell_dl);

            let t_min = *lg2_t.first().unwrap_or(&f64::NEG_INFINITY);
            let t_max = *lg2_t.last().unwrap_or(&f64::NEG_INFINITY);
            let domega_val = all_dcos[j] * dphis[phi_idx];

            (lg2_t, lg2_dl, domega_val, t_min, t_max)
        })
        .collect();

    // Unpack
    let mut lg2_t_obs = Vec::with_capacity(cells.len());
    let mut lg2_dl_domega = Vec::with_capacity(cells.len());
    let mut domega = Vec::with_capacity(cells.len());
    let mut lg2_t_min = Vec::with_capacity(cells.len());
    let mut lg2_t_max = Vec::with_capacity(cells.len());
    for (t, dl, d, tmin, tmax) in cells {
        if t.is_empty() { continue; }
        lg2_t_obs.push(t);
        lg2_dl_domega.push(dl);
        domega.push(d);
        lg2_t_min.push(tmin);
        lg2_t_max.push(tmax);
    }

    ForwardGrid::from_parts(lg2_t_obs, lg2_dl_domega, domega, lg2_t_min, lg2_t_max)
}

/// Convert a slice of f64 to log₂ representation.
fn to_lg2(vals: &[f64]) -> Vec<f64> {
    vals.iter().map(|&v| {
        if v > 0.0 { v.log2() } else { f64::NEG_INFINITY }
    }).collect()
}
