/// Representation of a 5-point interval for integration.
struct Interval {
    x1: f64,
    y: [f64; 5],
    h: f64,
    lorder: f64,
    horder: f64,
}

impl Interval {
    fn integrate(&mut self) {
        // Simpson's rule (low order)
        self.lorder = (self.y[0] + 4.0 * self.y[2] + self.y[4]) * self.h / 6.0;
        // Boole's rule (high order)
        self.horder = (7.0 * self.y[0] + 32.0 * self.y[1] + 12.0 * self.y[2]
            + 32.0 * self.y[3] + 7.0 * self.y[4])
            * self.h
            / 90.0;
    }
}

/// Adaptive 1D integral with initial x samples.
pub fn adaptive_1d<F: FnMut(f64) -> f64>(
    f: &mut F,
    xini: &[f64],
    xtol: f64,
    rtol: f64,
    max_iter: usize,
    force_return: bool,
) -> Result<f64, String> {
    let xsize = xini.len();
    let mut intervals: Vec<Interval> = Vec::new();
    let mut integral_tot = 0.0;

    for i in 0..xsize - 1 {
        let h = xini[i + 1] - xini[i];
        let x1 = xini[i];
        let y = [
            f(xini[i]),
            f(x1 + h / 4.0),
            f(x1 + 2.0 * h / 4.0),
            f(x1 + 3.0 * h / 4.0),
            f(xini[i + 1]),
        ];
        let mut itv = Interval {
            x1,
            y,
            h,
            lorder: 0.0,
            horder: 0.0,
        };
        itv.integrate();
        integral_tot += itv.horder;
        intervals.push(itv);
    }
    let h_tot = xini[xsize - 1] - xini[0];

    for _ in 0..max_iter {
        let mut any_refined = false;
        let intervals_size = intervals.len();
        let mut integral_new = integral_tot;

        // We need to iterate over existing intervals and possibly add new ones.
        // To avoid borrow issues, collect new intervals separately.
        let mut new_intervals: Vec<Interval> = Vec::new();

        for i in 0..intervals_size {
            let itv = &intervals[i];
            let need_refine = (itv.lorder - itv.horder).abs()
                > xtol * itv.h / h_tot + rtol * itv.horder.abs()
                && (itv.lorder - itv.horder).abs() * intervals_size as f64
                    > xtol + rtol * integral_tot.abs();

            if need_refine {
                integral_new -= itv.horder;

                // Right half
                let new_x1 = itv.x1 + itv.h / 2.0;
                let new_h = itv.h / 2.0;
                let new_y = [
                    itv.y[2],
                    f(new_x1 + new_h / 4.0),
                    itv.y[3],
                    f(new_x1 + 3.0 * new_h / 4.0),
                    itv.y[4],
                ];
                let mut new_itv = Interval {
                    x1: new_x1,
                    y: new_y,
                    h: new_h,
                    lorder: 0.0,
                    horder: 0.0,
                };
                new_itv.integrate();

                // Left half (modify in place)
                let left_h = itv.h / 2.0;
                let left_y = [
                    itv.y[0],
                    f(itv.x1 + left_h / 4.0),
                    itv.y[1],
                    f(itv.x1 + 3.0 * left_h / 4.0),
                    itv.y[2],
                ];
                let mut left_itv = Interval {
                    x1: itv.x1,
                    y: left_y,
                    h: left_h,
                    lorder: 0.0,
                    horder: 0.0,
                };
                left_itv.integrate();

                integral_new += new_itv.horder + left_itv.horder;
                any_refined = true;

                // Replace existing interval with left half
                intervals[i] = left_itv;
                new_intervals.push(new_itv);
            }
        }

        intervals.extend(new_intervals);
        integral_tot = integral_new;

        if !any_refined {
            return Ok(integral_tot);
        }
    }

    if force_return {
        Ok(integral_tot)
    } else {
        Err(format!(
            "Adaptive integration: convergence NOT achieved after {} iterations!",
            max_iter
        ))
    }
}

/// Adaptive 2D integral with initial x and y samples.
pub fn adaptive_2d<F: FnMut(f64, f64) -> f64>(
    f: &mut F,
    xini: &[f64],
    yini: &[f64],
    xtol: f64,
    rtol: f64,
    max_iter: usize,
    force_return: bool,
) -> Result<f64, String> {
    let xini_owned = xini.to_vec();
    let mut g = |y: f64| -> f64 {
        let mut h = |x: f64| -> f64 { f(x, y) };
        adaptive_1d(&mut h, &xini_owned, xtol, rtol, max_iter, force_return).unwrap_or(0.0)
    };
    adaptive_1d(&mut g, yini, xtol, rtol, max_iter, force_return)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        // integral of f(x) = 1 from 0 to 5 = 5
        let xini = vec![0.0, 5.0];
        let result = adaptive_1d(&mut |_x| 1.0, &xini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_linear() {
        // integral of f(x) = x from 0 to 4 = 8
        let xini = vec![0.0, 4.0];
        let result = adaptive_1d(&mut |x| x, &xini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 8.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_quadratic() {
        // integral of f(x) = x^2 from 0 to 3 = 9
        let xini = vec![0.0, 3.0];
        let result = adaptive_1d(&mut |x| x * x, &xini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 9.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_sin() {
        // integral of sin(x) from 0 to pi = 2
        let pi = std::f64::consts::PI;
        let xini = vec![0.0, pi / 2.0, pi];
        let result = adaptive_1d(&mut |x| x.sin(), &xini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_2d() {
        // integral of f(x,y) = 1 over [0,2] x [0,3] = 6
        let xini = vec![0.0, 2.0];
        let yini = vec![0.0, 3.0];
        let result = adaptive_2d(&mut |_x, _y| 1.0, &xini, &yini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 6.0).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_2d_product() {
        // integral of f(x,y) = x*y over [0,1] x [0,1] = 0.25
        let xini = vec![0.0, 1.0];
        let yini = vec![0.0, 1.0];
        let result = adaptive_2d(&mut |x, y| x * y, &xini, &yini, 1e-10, 1e-10, 50, true).unwrap();
        assert!((result - 0.25).abs() < 1e-8);
    }
}
