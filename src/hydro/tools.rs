use crate::math::root::brentq;

/// Utility class matching C++ Tool.
pub struct Tool {
    nwind: f64,
    nism: f64,
    rtol: f64,
    cal_level: i32,
    factor: f64,
}

impl Tool {
    pub fn new(nwind: f64, nism: f64, rtol: f64, cal_level: i32) -> Self {
        Tool {
            nwind,
            nism,
            rtol,
            cal_level,
            factor: 2.0,
        }
    }

    pub fn solve_density(&self, r: f64) -> f64 {
        let r17 = r / 1e17;
        self.nwind / (r17 * r17) + self.nism
    }

    pub fn solve_s(&self, r: f64, beta_gamma_sq: f64) -> f64 {
        let r17 = r / 1e17;
        let k = 2.0 * self.nwind / (self.nwind + self.nism * r17 * r17);

        match self.cal_level {
            0 => 1.0,
            1 => 0.52935729 - 0.05698377 * k - 0.00158176 * k * k - 0.00939548 * k * k * k,
            2 => {
                let s_bm =
                    0.52935729 - 0.05698377 * k - 0.00158176 * k * k - 0.00939548 * k * k * k;
                let s_st = 1.635 - 0.651 * k;
                (s_st + s_bm * self.factor * beta_gamma_sq)
                    / (1.0 + self.factor * beta_gamma_sq)
            }
            _ => panic!("Hydro: invalid calibration level!"),
        }
    }

    pub fn solve_beta_gamma_sq(&self, msw_eb: f64, mej_eb: f64, r: f64) -> Result<f64, String> {
        let beta_gamma_min_sq = 0.0;
        let gamma_max = 1.0 / (msw_eb + mej_eb);
        let beta_gamma_max_sq = (gamma_max * gamma_max - 1.0) * 1.01;

        let mut f = |u_sq: f64| -> f64 {
            let beta_sq = u_sq / (u_sq + 1.0);
            let gamma = (u_sq + 1.0).sqrt();
            let s = self.solve_s(r, u_sq);
            s * gamma * gamma * (1.0 + beta_sq * beta_sq / 3.0) * msw_eb
                + gamma * ((1.0 - s) * msw_eb + mej_eb)
                - 1.0
        };

        brentq(&mut f, beta_gamma_min_sq, beta_gamma_max_sq, 0.0, self.rtol, 100)
    }

    pub fn minmod(&self, x1: f64, x2: f64) -> f64 {
        if x1 * x2 > 0.0 {
            if x1.abs() < x2.abs() {
                x1
            } else {
                x2
            }
        } else {
            0.0
        }
    }

    /// Binary search for index range in a sorted array.
    pub fn find_index(&self, x_array: &[f64], x: f64) -> (usize, usize) {
        let mut index1 = 0usize;
        let mut index2 = x_array.len() - 1;

        while index2 - index1 > 1 {
            let index_mid = (index1 + index2) / 2;
            if x > x_array[index_mid] {
                index1 = index_mid;
            } else {
                index2 = index_mid;
            }
        }
        (index1, index2)
    }

    pub fn linear(&self, x: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
        if x1 == x2 {
            y1
        } else {
            (y2 - y1) / (x2 - x1) * (x - x1) + y1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> Tool {
        Tool::new(0.0, 1.0, 1e-6, 1)
    }

    #[test]
    fn test_solve_density_ism_only() {
        let t = Tool::new(0.0, 5.0, 1e-6, 1);
        assert!((t.solve_density(1e18) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_density_wind_only() {
        let t = Tool::new(1.0, 0.0, 1e-6, 1);
        // At r = 1e17: nwind / (1e17/1e17)^2 = nwind
        assert!((t.solve_density(1e17) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_density_combined() {
        let t = Tool::new(2.0, 3.0, 1e-6, 1);
        // At r = 1e17: 2/(1)^2 + 3 = 5
        assert!((t.solve_density(1e17) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_s_cal_level_0() {
        let t = Tool::new(0.0, 1.0, 1e-6, 0);
        assert!((t.solve_s(1e18, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_s_cal_level_1_ism() {
        // With nwind=0, k = 0, so s = 0.52935729
        let t = Tool::new(0.0, 1.0, 1e-6, 1);
        assert!((t.solve_s(1e18, 100.0) - 0.52935729).abs() < 1e-6);
    }

    #[test]
    fn test_minmod_same_sign() {
        let t = tool();
        assert!((t.minmod(3.0, 5.0) - 3.0).abs() < 1e-10);
        assert!((t.minmod(-5.0, -2.0) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_opposite_sign() {
        let t = tool();
        assert!((t.minmod(3.0, -5.0)).abs() < 1e-10);
        assert!((t.minmod(-3.0, 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_find_index() {
        let t = tool();
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(t.find_index(&arr, 2.5), (1, 2));
        assert_eq!(t.find_index(&arr, 4.5), (3, 4));
        assert_eq!(t.find_index(&arr, 1.5), (0, 1));
    }

    #[test]
    fn test_linear_interpolation() {
        let t = tool();
        // Midpoint between (0, 0) and (10, 20) at x=5 should give 10
        assert!((t.linear(5.0, 0.0, 10.0, 0.0, 20.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_at_endpoints() {
        let t = tool();
        assert!((t.linear(0.0, 0.0, 10.0, 3.0, 7.0) - 3.0).abs() < 1e-10);
        assert!((t.linear(10.0, 0.0, 10.0, 3.0, 7.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_degenerate() {
        let t = tool();
        // x1 == x2: should return y1
        assert!((t.linear(5.0, 5.0, 5.0, 3.0, 7.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_beta_gamma_sq() {
        // For a relativistic case, verify the root satisfies energy conservation
        let t = Tool::new(0.0, 1.0, 1e-6, 1);
        let msw_eb = 0.01;
        let mej_eb = 0.001;
        let r = 1e18;
        let u_sq = t.solve_beta_gamma_sq(msw_eb, mej_eb, r).unwrap();
        assert!(u_sq > 0.0);
        assert!(u_sq.is_finite());

        // Verify the root satisfies the equation f(u_sq) ~ 0
        let beta_sq = u_sq / (u_sq + 1.0);
        let gamma = (u_sq + 1.0).sqrt();
        let s = t.solve_s(r, u_sq);
        let residual = s * gamma * gamma * (1.0 + beta_sq * beta_sq / 3.0) * msw_eb
            + gamma * ((1.0 - s) * msw_eb + mej_eb)
            - 1.0;
        assert!(residual.abs() < 1e-5);
    }
}
