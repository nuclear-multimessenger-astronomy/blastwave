use crate::constants::*;
use crate::math::root::brentq;
use crate::math::optimize::{minimization, minimization_range};
use crate::math::integral::{adaptive_1d, adaptive_2d};
use crate::hydro::tools::Tool;
use crate::afterglow::blast::Blast;
use crate::afterglow::eats::EATS;
use crate::afterglow::models::{Dict, RadiationModel};

/// Afterglow flux density pipeline.
pub struct Afterglow {
    pub param: Dict,
    pub theta_v: f64,
    pub z: f64,
    pub d: f64,
    pub radiation_model: Option<RadiationModel>,
    pub avg_model: Option<RadiationModel>,
}

impl Afterglow {
    pub fn new() -> Self {
        Afterglow {
            param: Dict::new(),
            theta_v: 0.0,
            z: 0.0,
            d: 0.0,
            radiation_model: None,
            avg_model: None,
        }
    }

    pub fn config_parameters(&mut self, param: Dict) {
        self.theta_v = *param
            .get("theta_v")
            .expect("Mandatory parameter 'theta_v' missing");
        self.d = *param.get("d").expect("Mandatory parameter 'd' missing");
        self.z = *param.get("z").expect("Mandatory parameter 'z' missing");
        self.param = param;
    }

    pub fn config_intensity(&mut self, model_name: &str) {
        self.radiation_model = Some(
            crate::afterglow::models::get_radiation_model(model_name)
                .unwrap_or_else(|| panic!("No such radiation model: '{}'", model_name)),
        );
    }

    pub fn config_avg_model(&mut self, model_name: &str) {
        self.avg_model = Some(
            crate::afterglow::models::get_avg_model(model_name)
                .unwrap_or_else(|| panic!("No such weighted average model: '{}'", model_name)),
        );
    }

    pub fn intensity(
        &self,
        tobs: f64,
        nu: f64,
        theta: f64,
        phi: f64,
        eats: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let tobs_z = tobs / (1.0 + self.z);
        let nu_z = nu * (1.0 + self.z);

        let mut blast = Blast::default();
        if !eats.solve_blast(
            tobs_z, theta, phi, self.theta_v, y_data, t_data, theta_data, tool, &mut blast,
        ) {
            return 0.0;
        }

        let nu_src = nu_z / blast.doppler;
        let model = self.radiation_model.unwrap();
        let intensity = model(nu_src, &self.param, &blast);

        intensity / 4.0 / PI * blast.doppler * blast.doppler * blast.doppler
    }

    fn dl_domega(
        &self,
        nu_z: f64,
        blast: &Blast,
    ) -> f64 {
        let nu_src = nu_z / blast.doppler;
        let model = self.radiation_model.unwrap();
        let intensity = model(nu_src, &self.param, blast);
        intensity * blast.r * blast.r * blast.doppler * blast.doppler * blast.doppler
    }

    fn find_peak(
        &self,
        tobs_z: f64,
        nu_z: f64,
        eats: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let mut f = |theta: f64| -> f64 {
            let mut blast = Blast::default();
            if !eats.solve_blast(
                tobs_z, theta, 0.0, self.theta_v, y_data, t_data, theta_data, tool, &mut blast,
            ) {
                return 0.0; // no contribution from out-of-range times
            }
            -self.dl_domega(nu_z, &blast)
        };

        minimization(&mut f, theta_data, 1e-6, 100)
    }

    pub fn luminosity(
        &self,
        tobs: f64,
        nu: f64,
        rtol: f64,
        max_iter: usize,
        force_return: bool,
        eats: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let tobs_z = tobs / (1.0 + self.z);
        let nu_z = nu * (1.0 + self.z);
        let theta_peak = self.find_peak(tobs_z, nu_z, eats, y_data, t_data, theta_data, tool);

        // Get beaming angle
        let mut blast_peak = Blast::default();
        if !eats.solve_blast(
            tobs_z,
            theta_peak,
            0.0,
            self.theta_v,
            y_data,
            t_data,
            theta_data,
            tool,
            &mut blast_peak,
        ) {
            return 0.0; // tobs exceeds PDE max
        }
        let beaming_angle = 1.0 / blast_peak.gamma;

        let mut f = |cos_theta_rot: f64, phi_rot: f64| -> f64 {
            let theta_rot = cos_theta_rot.acos();

            let x = theta_rot.sin() * phi_rot.cos() * theta_peak.cos()
                + theta_rot.cos() * theta_peak.sin();
            let y = theta_rot.sin() * phi_rot.sin();
            let z_val = -theta_rot.sin() * phi_rot.cos() * theta_peak.sin()
                + theta_rot.cos() * theta_peak.cos();

            let cos_theta = z_val.abs().min(1.0) * z_val.signum();
            let theta = cos_theta.acos();

            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += PI * 2.0;
            }

            let mut blast = Blast::default();
            if !eats.solve_blast(
                tobs_z, theta, phi, self.theta_v, y_data, t_data, theta_data, tool, &mut blast,
            ) {
                return 0.0;
            }
            self.dl_domega(nu_z, &blast)
        };

        // Build initial sample grid dense enough to capture the narrow emission beam.
        // The beaming angle 1/gamma can be very small, so we place samples both
        // around the beam and uniformly across the domain to avoid the adaptive
        // integrator seeing all-zeros and skipping refinement.
        let mut cos_theta_samples = vec![-1.0];
        // Uniform samples to cover the full domain
        let n_uniform = 8;
        for i in 1..n_uniform {
            cos_theta_samples.push(-1.0 + 2.0 * i as f64 / n_uniform as f64);
        }
        // Targeted samples around the beaming cone
        let ba = beaming_angle;
        for &frac in &[0.25, 0.5, 1.0, 2.0] {
            let val = (ba * frac).cos();
            cos_theta_samples.push(val);
        }
        cos_theta_samples.push(1.0);
        // Sort and deduplicate (required by adaptive_1d)
        cos_theta_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cos_theta_samples.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

        let phi_samples = vec![0.0, PI / 2.0, PI];

        let luminosity = adaptive_2d(
            &mut f,
            &cos_theta_samples,
            &phi_samples,
            0.0,
            rtol,
            max_iter,
            force_return,
        )
        .unwrap_or(0.0)
            * 2.0;

        luminosity
    }

    pub fn freq_int_l(
        &self,
        tobs: f64,
        nu1: f64,
        nu2: f64,
        rtol: f64,
        max_iter: usize,
        force_return: bool,
        eats: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let mut f = |nu: f64| -> f64 {
            self.luminosity(tobs, nu, rtol, max_iter, force_return, eats, y_data, t_data, theta_data, tool)
        };
        adaptive_1d(&mut f, &[nu1, nu2], 0.0, rtol, max_iter, force_return).unwrap_or(0.0)
    }

    pub fn integrate_model(
        &self,
        tobs: f64,
        nu: f64,
        rtol: f64,
        max_iter: usize,
        force_return: bool,
        eats: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let tobs_z = tobs / (1.0 + self.z);
        let nu_z = nu * (1.0 + self.z);
        let theta_peak = self.find_peak(tobs_z, nu_z, eats, y_data, t_data, theta_data, tool);

        // Get beaming angle & nu_src from peak blast
        let mut blast_peak = Blast::default();
        if !eats.solve_blast(
            tobs_z,
            theta_peak,
            0.0,
            self.theta_v,
            y_data,
            t_data,
            theta_data,
            tool,
            &mut blast_peak,
        ) {
            return 0.0; // tobs exceeds PDE max
        }
        let beaming_angle = 1.0 / blast_peak.gamma;
        let nu_src = nu_z / blast_peak.doppler;
        let avg_model = self.avg_model.unwrap();

        let mut f = |cos_theta_rot: f64, phi_rot: f64| -> f64 {
            let theta_rot = cos_theta_rot.acos();

            let x = theta_rot.sin() * phi_rot.cos() * theta_peak.cos()
                + theta_rot.cos() * theta_peak.sin();
            let y = theta_rot.sin() * phi_rot.sin();
            let z_val = -theta_rot.sin() * phi_rot.cos() * theta_peak.sin()
                + theta_rot.cos() * theta_peak.cos();

            let cos_theta = z_val.abs().min(1.0) * z_val.signum();
            let theta = cos_theta.acos();

            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += PI * 2.0;
            }

            let mut blast = Blast::default();
            if !eats.solve_blast(
                tobs_z, theta, phi, self.theta_v, y_data, t_data, theta_data, tool, &mut blast,
            ) {
                return 0.0;
            }
            avg_model(nu_src, &self.param, &blast) * self.dl_domega(nu_z, &blast)
        };

        let cos_theta_samples = vec![
            -1.0,
            beaming_angle.cos(),
            (beaming_angle / 2.0).cos(),
            1.0,
        ];
        let phi_samples = vec![0.0, PI, 2.0 * PI];

        adaptive_2d(
            &mut f,
            &cos_theta_samples,
            &phi_samples,
            0.0,
            rtol,
            max_iter,
            force_return,
        )
        .unwrap_or(0.0)
    }

    pub fn intensity_of_pixel(
        &self,
        tobs: f64,
        nu: f64,
        x_offset: f64,
        y_offset: f64,
        eats_solver: &EATS,
        y_data: &[Vec<Vec<f64>>],
        t_data: &[f64],
        theta_data: &[f64],
        tool: &Tool,
    ) -> f64 {
        let theta_v = self.theta_v;
        let z = self.z;
        let d = self.d;

        let projection =
            (x_offset * x_offset + y_offset * y_offset).sqrt() * d * MPC * (1.0 + z) * (1.0 + z) * MAS;
        let phi_tilde = {
            let p = y_offset.atan2(x_offset);
            if p < 0.0 { p + PI * 2.0 } else { p }
        };

        let f_root = |theta_tilde: f64| -> f64 {
            let x = theta_tilde.sin() * phi_tilde.cos() * theta_v.cos()
                + theta_tilde.cos() * theta_v.sin();
            let y = theta_tilde.sin() * phi_tilde.sin();
            let z_val = -theta_tilde.sin() * phi_tilde.cos() * theta_v.sin()
                + theta_tilde.cos() * theta_v.cos();
            let theta = z_val.acos();
            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += PI * 2.0;
            }

            let mut blast = Blast::default();
            if !eats_solver.solve_blast(
                tobs / (1.0 + z),
                theta,
                phi,
                theta_v,
                y_data,
                t_data,
                theta_data,
                tool,
                &mut blast,
            ) {
                return projection; // treat as no intersection when out of range
            }
            projection - blast.r * theta_tilde.sin()
        };

        let theta_tilde_peak =
            minimization_range(&mut |x| f_root(x), 0.0, PI, 10, 1e-4, "linear", 100).unwrap_or(PI / 2.0);

        let mut intensity_tot = 0.0;

        let f1 = f_root(0.0);
        let f2 = f_root(theta_tilde_peak);
        let f3 = f_root(PI);

        if f1 * f2 <= 0.0 {
            if let Ok(theta_tilde) = brentq(&mut |x| f_root(x), 0.0, theta_tilde_peak, 1e-6, 1e-6, 100) {
                intensity_tot += self.intensity(tobs, nu, {
                    let x = theta_tilde.sin() * phi_tilde.cos() * theta_v.cos()
                        + theta_tilde.cos() * theta_v.sin();
                    let y = theta_tilde.sin() * phi_tilde.sin();
                    let z_val = -theta_tilde.sin() * phi_tilde.cos() * theta_v.sin()
                        + theta_tilde.cos() * theta_v.cos();
                    let theta = z_val.acos();
                    theta
                }, {
                    let x = theta_tilde.sin() * phi_tilde.cos() * theta_v.cos()
                        + theta_tilde.cos() * theta_v.sin();
                    let y = theta_tilde.sin() * phi_tilde.sin();
                    let mut phi = y.atan2(x);
                    if phi < 0.0 { phi += PI * 2.0; }
                    phi
                }, eats_solver, y_data, t_data, theta_data, tool);
            }
        }
        if f2 * f3 <= 0.0 {
            if let Ok(theta_tilde) = brentq(&mut |x| f_root(x), theta_tilde_peak, PI, 1e-6, 1e-6, 100) {
                intensity_tot += self.intensity(tobs, nu, {
                    let x = theta_tilde.sin() * phi_tilde.cos() * theta_v.cos()
                        + theta_tilde.cos() * theta_v.sin();
                    let y = theta_tilde.sin() * phi_tilde.sin();
                    let z_val = -theta_tilde.sin() * phi_tilde.cos() * theta_v.sin()
                        + theta_tilde.cos() * theta_v.cos();
                    z_val.acos()
                }, {
                    let x = theta_tilde.sin() * phi_tilde.cos() * theta_v.cos()
                        + theta_tilde.cos() * theta_v.sin();
                    let y = theta_tilde.sin() * phi_tilde.sin();
                    let mut phi = y.atan2(x);
                    if phi < 0.0 { phi += PI * 2.0; }
                    phi
                }, eats_solver, y_data, t_data, theta_data, tool);
            }
        }

        intensity_tot
    }
}
