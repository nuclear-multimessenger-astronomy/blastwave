#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1};
use std::collections::HashMap;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Post-processing: interpolate isolated zero-valued luminosity points
// ---------------------------------------------------------------------------

/// Log-log interpolate zero runs in a time-sorted, single-frequency group.
/// Zeros bounded on both sides by positive values are filled via log-log
/// interpolation (natural for power-law lightcurves).  Zeros at the
/// boundaries (leading/trailing) are left untouched.
fn interpolate_sorted_group(results: &mut [f64], times: &[f64]) {
    let n = results.len();
    let mut i = 0;
    while i < n {
        if results[i] == 0.0 {
            let start = i;
            while i < n && results[i] == 0.0 {
                i += 1;
            }
            let end = i; // exclusive

            let has_left = start > 0 && results[start - 1] > 0.0;
            let has_right = end < n && results[end] > 0.0;

            if has_left && has_right {
                let li = start - 1;
                let ri = end;
                let log_lum_l = results[li].ln();
                let log_lum_r = results[ri].ln();
                let log_t_l = times[li].ln();
                let log_t_r = times[ri].ln();
                let dt = log_t_r - log_t_l;

                if dt > 0.0 {
                    for j in start..end {
                        let frac = (times[j].ln() - log_t_l) / dt;
                        results[j] = (log_lum_l + frac * (log_lum_r - log_lum_l)).exp();
                    }
                }
            }
        } else {
            i += 1;
        }
    }
}

/// Interpolate zero-valued luminosity entries that are surrounded by
/// non-zero values at the same observing frequency.  Works for the common
/// single-frequency case (fast path) as well as mixed-frequency arrays.
fn interpolate_zero_luminosities(results: &mut [f64], t_s: &[f64], nu_s: &[f64]) {
    let n = results.len();
    if n < 3 {
        return;
    }

    // Fast path: all entries at the same frequency
    let all_same_nu = nu_s.iter().all(|&v| v == nu_s[0]);

    if all_same_nu {
        let sorted = t_s.windows(2).all(|w| w[0] <= w[1]);
        if sorted {
            interpolate_sorted_group(results, t_s);
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| t_s[a].partial_cmp(&t_s[b]).unwrap());
            let mut sorted_res: Vec<f64> = order.iter().map(|&i| results[i]).collect();
            let sorted_t: Vec<f64> = order.iter().map(|&i| t_s[i]).collect();
            interpolate_sorted_group(&mut sorted_res, &sorted_t);
            for (j, &idx) in order.iter().enumerate() {
                results[idx] = sorted_res[j];
            }
        }
    } else {
        // Multi-frequency: group by nu (using bit-exact equality), process each
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for i in 0..n {
            groups.entry(nu_s[i].to_bits()).or_default().push(i);
        }
        for (_, mut indices) in groups {
            if indices.len() < 3 {
                continue;
            }
            indices.sort_by(|&a, &b| t_s[a].partial_cmp(&t_s[b]).unwrap());
            let mut group_res: Vec<f64> = indices.iter().map(|&i| results[i]).collect();
            let group_t: Vec<f64> = indices.iter().map(|&i| t_s[i]).collect();
            interpolate_sorted_group(&mut group_res, &group_t);
            for (j, &idx) in indices.iter().enumerate() {
                results[idx] = group_res[j];
            }
        }
    }
}

use jetsimpy_rs::constants::*;
use jetsimpy_rs::hydro::config::JetConfig;
use jetsimpy_rs::hydro::sim_box::SimBox;
use jetsimpy_rs::hydro::interpolate::Interpolator;
use jetsimpy_rs::hydro::tools::Tool;
use jetsimpy_rs::afterglow::blast::Blast;
use jetsimpy_rs::afterglow::eats::EATS;
use jetsimpy_rs::afterglow::afterglow::Afterglow;

#[pymodule]
fn jetsimpy_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJetConfig>()?;
    m.add_class::<PyJet>()?;
    m.add_class::<PyBlast>()?;
    Ok(())
}

/// Python-exposed JetConfig.
#[pyclass(name = "JetConfig")]
pub struct PyJetConfig {
    #[pyo3(get, set)]
    pub theta_edge: Vec<f64>,
    #[pyo3(get, set, name = "Eb")]
    pub eb: Vec<f64>,
    #[pyo3(get, set, name = "Ht")]
    pub ht: Vec<f64>,
    #[pyo3(get, set, name = "Msw")]
    pub msw: Vec<f64>,
    #[pyo3(get, set, name = "Mej")]
    pub mej: Vec<f64>,
    #[pyo3(get, set, name = "R")]
    pub r: Vec<f64>,
    #[pyo3(get, set)]
    pub nwind: f64,
    #[pyo3(get, set)]
    pub nism: f64,
    #[pyo3(get, set)]
    pub tmin: f64,
    #[pyo3(get, set)]
    pub tmax: f64,
    #[pyo3(get, set)]
    pub rtol: f64,
    #[pyo3(get, set)]
    pub cfl: f64,
    #[pyo3(get, set)]
    pub spread: bool,
    #[pyo3(get, set)]
    pub cal_level: i32,
}

#[pymethods]
impl PyJetConfig {
    #[new]
    fn new() -> Self {
        PyJetConfig {
            theta_edge: Vec::new(),
            eb: Vec::new(),
            ht: Vec::new(),
            msw: Vec::new(),
            mej: Vec::new(),
            r: Vec::new(),
            nwind: 0.0,
            nism: 0.0,
            tmin: 10.0,
            tmax: 1e10,
            rtol: 1e-6,
            cfl: 0.9,
            spread: true,
            cal_level: 1,
        }
    }
}

impl PyJetConfig {
    fn to_config(&self) -> JetConfig {
        JetConfig {
            theta_edge: self.theta_edge.clone(),
            eb: self.eb.clone(),
            ht: self.ht.clone(),
            msw: self.msw.clone(),
            mej: self.mej.clone(),
            r: self.r.clone(),
            nwind: self.nwind,
            nism: self.nism,
            tmin: self.tmin,
            tmax: self.tmax,
            rtol: self.rtol,
            cfl: self.cfl,
            spread: self.spread,
            cal_level: self.cal_level,
        }
    }
}

/// Inner data structure that is Send + Sync for rayon.
struct JetInner {
    ys: Vec<Vec<Vec<f64>>>,
    ts: Vec<f64>,
    theta: Vec<f64>,
    tool: Tool,
    interpolator: Interpolator,
    eats: EATS,
    afterglow: Afterglow,
}

// SAFETY: JetInner only contains owned, immutable-after-construction data.
// The `afterglow` field is mutated only through &mut self on PyJet (which holds the GIL).
unsafe impl Send for JetInner {}
unsafe impl Sync for JetInner {}

/// Python-exposed Jet class.
#[pyclass(name = "Jet")]
pub struct PyJet {
    inner: Option<JetInner>,
    config: JetConfig,
}

#[pymethods]
impl PyJet {
    #[new]
    fn new(config: &PyJetConfig) -> Self {
        PyJet {
            inner: None,
            config: config.to_config(),
        }
    }

    fn solveJet(&mut self) -> PyResult<()> {
        let mut sim_box = SimBox::new(&self.config);
        sim_box.solve_pde();

        let theta = sim_box.get_theta().clone();
        let ts = sim_box.ts.clone();
        let ys = sim_box.ys.clone();
        let tool = Tool::new(
            self.config.nwind,
            self.config.nism,
            self.config.rtol,
            self.config.cal_level,
        );

        let interpolator = Interpolator::new(&theta, &ts);
        let eats = EATS::new(&theta, &ts);
        let afterglow = Afterglow::new();

        self.inner = Some(JetInner {
            ys,
            ts,
            theta,
            tool,
            interpolator,
            eats,
            afterglow,
        });
        Ok(())
    }

    fn getY<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        let nvar = 5;
        let ntheta = inner.theta.len();
        let nt = inner.ts.len();

        let arr = PyArray3::<f64>::zeros(py, [nvar, ntheta, nt], false);
        {
            let mut arr_rw = unsafe { arr.as_array_mut() };
            for i in 0..nvar {
                for j in 0..ntheta {
                    for k in 0..nt {
                        arr_rw[[i, j, k]] = inner.ys[i][j][k];
                    }
                }
            }
        }
        Ok(arr)
    }

    fn getT<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        Ok(PyArray1::from_slice(py, &inner.ts))
    }

    fn getTheta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        Ok(PyArray1::from_slice(py, &inner.theta))
    }

    // ---------- Hydro interpolation ----------

    fn interpolateMsw(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        self.vectorized_interpolate(t, theta, 0, false, py)
    }

    fn interpolateMej(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        self.vectorized_interpolate(t, theta, 1, false, py)
    }

    fn interpolateBetaGamma(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        self.vectorized_interpolate(t, theta, 2, true, py)
    }

    fn interpolateBetaTh(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        self.vectorized_interpolate(t, theta, 3, false, py)
    }

    fn interpolateR(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        self.vectorized_interpolate(t, theta, 4, false, py)
    }

    fn interpolateE0(&self, t: PyObject, theta: PyObject, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        if let (Ok(t_arr), Ok(th_arr)) = (
            t.extract::<PyReadonlyArray1<f64>>(py),
            theta.extract::<PyReadonlyArray1<f64>>(py),
        ) {
            let t_slice = t_arr.as_slice()?;
            let th_slice = th_arr.as_slice()?;
            let n = t_slice.len();
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                result[i] = self.compute_e0(inner, t_slice[i], th_slice[i]);
            }
            Ok(PyArray1::from_vec(py, result).into_any().unbind())
        } else {
            let t_val: f64 = t.extract(py)?;
            let th_val: f64 = theta.extract(py)?;
            let result = self.compute_e0(inner, t_val, th_val);
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    // ---------- Afterglow configuration ----------

    fn configParameters(&mut self, param: HashMap<String, f64>) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        inner.afterglow.config_parameters(param);
        Ok(())
    }

    fn configIntensity(&mut self, model_name: &str) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        inner.afterglow.config_intensity(model_name);
        Ok(())
    }

    fn configAvgModel(&mut self, model_name: &str) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;
        inner.afterglow.config_avg_model(model_name);
        Ok(())
    }

    fn configIntensityPy(&mut self, _py_f: PyObject) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "Python callback radiation models are not supported in the Rust version. Use built-in models ('sync', 'sync_dnp')."
        ))
    }

    fn configAvgModelPy(&mut self, _py_f: PyObject) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "Python callback average models are not supported in the Rust version. Use built-in models."
        ))
    }

    // ---------- Afterglow calculations ----------

    fn calculateEATS(&self, tobs: PyObject, theta: PyObject, phi: PyObject, theta_v: PyObject, z: PyObject, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        let n = broadcast_len(&[&tobs, &theta, &phi, &theta_v, &z], py)?;
        if let Some(n) = n {
            let (t_s, _) = extract_broadcast(&tobs, py, Some(n))?;
            let (th_s, _) = extract_broadcast(&theta, py, Some(n))?;
            let (ph_s, _) = extract_broadcast(&phi, py, Some(n))?;
            let (tv_s, _) = extract_broadcast(&theta_v, py, Some(n))?;
            let (z_s, _) = extract_broadcast(&z, py, Some(n))?;
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                let tobs_z = t_s[i] / (1.0 + z_s[i]);
                result[i] = inner.eats.solve_eats(
                    tobs_z, th_s[i], ph_s[i], tv_s[i],
                    &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                );
            }
            Ok(PyArray1::from_vec(py, result).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let th_val: f64 = theta.extract(py)?;
            let ph_val: f64 = phi.extract(py)?;
            let tv_val: f64 = theta_v.extract(py)?;
            let z_val: f64 = z.extract(py)?;
            let tobs_z = t_val / (1.0 + z_val);
            let result = inner.eats.solve_eats(
                tobs_z, th_val, ph_val, tv_val,
                &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn calculateIntensity(&self, tobs: PyObject, nu: PyObject, theta: PyObject, phi: PyObject, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        let n = broadcast_len(&[&tobs, &nu, &theta, &phi], py)?;
        if let Some(n) = n {
            let (t_s, _) = extract_broadcast(&tobs, py, Some(n))?;
            let (nu_s, _) = extract_broadcast(&nu, py, Some(n))?;
            let (th_s, _) = extract_broadcast(&theta, py, Some(n))?;
            let (ph_s, _) = extract_broadcast(&phi, py, Some(n))?;
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                result[i] = inner.afterglow.intensity(
                    t_s[i], nu_s[i], th_s[i], ph_s[i],
                    &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                );
            }
            Ok(PyArray1::from_vec(py, result).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let nu_val: f64 = nu.extract(py)?;
            let th_val: f64 = theta.extract(py)?;
            let ph_val: f64 = phi.extract(py)?;
            let result = inner.afterglow.intensity(
                t_val, nu_val, th_val, ph_val,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (tobs, nu, rtol, max_iter=50, force_return=true))]
    fn calculateLuminosity(
        &self,
        tobs: PyObject,
        nu: PyObject,
        rtol: PyObject,
        max_iter: i32,
        force_return: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        let n = broadcast_len(&[&tobs, &nu, &rtol], py)?;
        if let Some(n) = n {
            let (t_s, _) = extract_broadcast(&tobs, py, Some(n))?;
            let (nu_s, _) = extract_broadcast(&nu, py, Some(n))?;
            let (rtol_s, _) = extract_broadcast(&rtol, py, Some(n))?;

            let mut results: Vec<f64> = py.allow_threads(|| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        inner.afterglow.luminosity(
                            t_s[i], nu_s[i], rtol_s[i],
                            max_iter as usize, force_return,
                            &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                        )
                    })
                    .collect()
            });

            // Fill isolated zero-valued points via log-log interpolation
            interpolate_zero_luminosities(&mut results, &t_s, &nu_s);

            Ok(PyArray1::from_vec(py, results).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let nu_val: f64 = nu.extract(py)?;
            let rtol_val: f64 = rtol.extract(py)?;
            let result = inner.afterglow.luminosity(
                t_val, nu_val, rtol_val,
                max_iter as usize, force_return,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (tobs, nu1, nu2, rtol, max_iter=50, force_return=true))]
    fn calculateFreqIntL(
        &self,
        tobs: PyObject,
        nu1: PyObject,
        nu2: PyObject,
        rtol: PyObject,
        max_iter: i32,
        force_return: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        let n = broadcast_len(&[&tobs, &nu1, &nu2, &rtol], py)?;
        if let Some(n) = n {
            let (t_s, _) = extract_broadcast(&tobs, py, Some(n))?;
            let (nu1_s, _) = extract_broadcast(&nu1, py, Some(n))?;
            let (nu2_s, _) = extract_broadcast(&nu2, py, Some(n))?;
            let (rtol_s, _) = extract_broadcast(&rtol, py, Some(n))?;

            let results: Vec<f64> = py.allow_threads(|| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        inner.afterglow.freq_int_l(
                            t_s[i], nu1_s[i], nu2_s[i], rtol_s[i],
                            max_iter as usize, force_return,
                            &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                        )
                    })
                    .collect()
            });
            Ok(PyArray1::from_vec(py, results).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let nu1_val: f64 = nu1.extract(py)?;
            let nu2_val: f64 = nu2.extract(py)?;
            let rtol_val: f64 = rtol.extract(py)?;
            let result = inner.afterglow.freq_int_l(
                t_val, nu1_val, nu2_val, rtol_val,
                max_iter as usize, force_return,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (tobs, nu, rtol, max_iter=50, force_return=true))]
    fn WeightedAverage(
        &self,
        tobs: PyObject,
        nu: PyObject,
        rtol: PyObject,
        max_iter: i32,
        force_return: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        let n = broadcast_len(&[&tobs, &nu, &rtol], py)?;
        if let Some(n) = n {
            let (t_s, _) = extract_broadcast(&tobs, py, Some(n))?;
            let (nu_s, _) = extract_broadcast(&nu, py, Some(n))?;
            let (rtol_s, _) = extract_broadcast(&rtol, py, Some(n))?;

            let results: Vec<f64> = py.allow_threads(|| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        if t_s[i] == 0.0 {
                            return 0.0;
                        }
                        let lum = inner.afterglow.luminosity(
                            t_s[i], nu_s[i], rtol_s[i],
                            max_iter as usize, force_return,
                            &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                        );
                        let integral = inner.afterglow.integrate_model(
                            t_s[i], nu_s[i], rtol_s[i],
                            max_iter as usize, force_return,
                            &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                        );
                        integral / lum
                    })
                    .collect()
            });
            Ok(PyArray1::from_vec(py, results).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let nu_val: f64 = nu.extract(py)?;
            let rtol_val: f64 = rtol.extract(py)?;
            if t_val == 0.0 {
                return Ok(0.0f64.into_pyobject(py)?.into_any().unbind());
            }
            let lum = inner.afterglow.luminosity(
                t_val, nu_val, rtol_val,
                max_iter as usize, force_return,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            let integral = inner.afterglow.integrate_model(
                t_val, nu_val, rtol_val,
                max_iter as usize, force_return,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok((integral / lum).into_pyobject(py)?.into_any().unbind())
        }
    }

    fn IntensityOfPixel(&self, tobs: PyObject, nu: PyObject, x_tilde: PyObject, y_tilde: PyObject, py: Python) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        if let (Ok(t_arr), Ok(nu_arr), Ok(x_arr), Ok(y_arr)) = (
            tobs.extract::<PyReadonlyArray1<f64>>(py),
            nu.extract::<PyReadonlyArray1<f64>>(py),
            x_tilde.extract::<PyReadonlyArray1<f64>>(py),
            y_tilde.extract::<PyReadonlyArray1<f64>>(py),
        ) {
            let t_s = t_arr.as_slice()?;
            let nu_s = nu_arr.as_slice()?;
            let x_s = x_arr.as_slice()?;
            let y_s = y_arr.as_slice()?;
            let n = t_s.len();
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                result[i] = inner.afterglow.intensity_of_pixel(
                    t_s[i], nu_s[i], x_s[i], y_s[i],
                    &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
                );
            }
            Ok(PyArray1::from_vec(py, result).into_any().unbind())
        } else {
            let t_val: f64 = tobs.extract(py)?;
            let nu_val: f64 = nu.extract(py)?;
            let x_val: f64 = x_tilde.extract(py)?;
            let y_val: f64 = y_tilde.extract(py)?;
            let result = inner.afterglow.intensity_of_pixel(
                t_val, nu_val, x_val, y_val,
                &inner.eats, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// Extract a PyObject as either a scalar (broadcast to len n) or an array.
fn extract_broadcast(obj: &PyObject, py: Python, n: Option<usize>) -> PyResult<(Vec<f64>, Option<usize>)> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>(py) {
        let slice = arr.as_slice()?.to_vec();
        let len = slice.len();
        if let Some(expected) = n {
            if len != expected {
                return Err(PyRuntimeError::new_err(format!(
                    "Array length mismatch: expected {}, got {}", expected, len
                )));
            }
        }
        Ok((slice, Some(len)))
    } else {
        let val: f64 = obj.extract(py)?;
        match n {
            Some(len) => Ok((vec![val; len], Some(len))),
            None => Ok((vec![val], None)),
        }
    }
}

/// Determine broadcast length from multiple PyObjects.
fn broadcast_len(objs: &[&PyObject], py: Python) -> PyResult<Option<usize>> {
    let mut n: Option<usize> = None;
    for obj in objs {
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>(py) {
            let len = arr.as_slice()?.len();
            if let Some(existing) = n {
                if len != existing {
                    return Err(PyRuntimeError::new_err(format!(
                        "Array length mismatch: {} vs {}", existing, len
                    )));
                }
            }
            n = Some(len);
        }
    }
    Ok(n)
}

impl PyJet {
    fn vectorized_interpolate(
        &self,
        t: PyObject,
        theta: PyObject,
        y_index: usize,
        sqrt: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("solveJet() must be called first")
        })?;

        if let (Ok(t_arr), Ok(th_arr)) = (
            t.extract::<PyReadonlyArray1<f64>>(py),
            theta.extract::<PyReadonlyArray1<f64>>(py),
        ) {
            let t_slice = t_arr.as_slice()?;
            let th_slice = th_arr.as_slice()?;
            let n = t_slice.len();
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                let mut val = inner.interpolator.interpolate_y(
                    t_slice[i],
                    th_slice[i],
                    y_index,
                    &inner.ys,
                    &inner.ts,
                    &inner.theta,
                    &inner.tool,
                );
                if sqrt {
                    val = val.sqrt();
                }
                result[i] = val;
            }
            Ok(PyArray1::from_vec(py, result).into_any().unbind())
        } else {
            let t_val: f64 = t.extract(py)?;
            let th_val: f64 = theta.extract(py)?;
            let mut val = inner.interpolator.interpolate_y(
                t_val, th_val, y_index, &inner.ys, &inner.ts, &inner.theta, &inner.tool,
            );
            if sqrt {
                val = val.sqrt();
            }
            Ok(val.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn compute_e0(&self, inner: &JetInner, t: f64, theta: f64) -> f64 {
        let msw = inner.interpolator.interpolate_y(t, theta, 0, &inner.ys, &inner.ts, &inner.theta, &inner.tool);
        let mej = inner.interpolator.interpolate_y(t, theta, 1, &inner.ys, &inner.ts, &inner.theta, &inner.tool);
        let beta_gamma_sq = inner.interpolator.interpolate_y(t, theta, 2, &inner.ys, &inner.ts, &inner.theta, &inner.tool);
        let r = inner.interpolator.interpolate_y(t, theta, 4, &inner.ys, &inner.ts, &inner.theta, &inner.tool);
        let s = inner.tool.solve_s(r, beta_gamma_sq);

        let e0 = s
            * (1.0
                + beta_gamma_sq * beta_gamma_sq / (beta_gamma_sq + 1.0) / (beta_gamma_sq + 1.0)
                    / 3.0)
            * (beta_gamma_sq + 1.0)
            * msw
            + (1.0 - s) * (beta_gamma_sq + 1.0).sqrt() * msw
            + (beta_gamma_sq + 1.0).sqrt() * mej
            - msw
            - mej;
        e0 * C_SPEED * C_SPEED
    }
}

/// Python-exposed Blast struct (read-only properties).
#[pyclass(name = "Blast")]
pub struct PyBlast {
    inner: Blast,
}

#[pymethods]
impl PyBlast {
    #[getter]
    fn t(&self) -> f64 { self.inner.t }
    #[getter]
    fn theta(&self) -> f64 { self.inner.theta }
    #[getter]
    fn phi(&self) -> f64 { self.inner.phi }
    #[getter]
    fn R(&self) -> f64 { self.inner.r }
    #[getter]
    fn beta(&self) -> f64 { self.inner.beta }
    #[getter]
    fn gamma(&self) -> f64 { self.inner.gamma }
    #[getter]
    fn beta_th(&self) -> f64 { self.inner.beta_th }
    #[getter]
    fn beta_r(&self) -> f64 { self.inner.beta_r }
    #[getter]
    fn beta_f(&self) -> f64 { self.inner.beta_f }
    #[getter]
    fn gamma_f(&self) -> f64 { self.inner.gamma_f }
    #[getter]
    fn s(&self) -> f64 { self.inner.s }
    #[getter]
    fn doppler(&self) -> f64 { self.inner.doppler }
    #[getter]
    fn n_blast(&self) -> f64 { self.inner.n_blast }
    #[getter]
    fn e_density(&self) -> f64 { self.inner.e_density }
    #[getter]
    fn pressure(&self) -> f64 { self.inner.pressure }
    #[getter]
    fn n_ambient(&self) -> f64 { self.inner.n_ambient }
    #[getter]
    fn dR(&self) -> f64 { self.inner.dr }
}
