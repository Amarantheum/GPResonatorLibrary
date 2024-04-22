use std::{fmt::{Debug, Display}, error::Error};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{DMatrix, DVector, Dyn, MatrixXx3, Owned, Vector3, U3};
use num_complex::Complex;

const BARRIER_SCALE_FACTOR: f64 = 1e0;

#[derive(Debug)]
pub struct ResonatorParams {
    pub w_0: f64,
    pub g: f64,
    pub r: f64,
}

impl ResonatorParams {
    #[inline]
    pub fn init(w_0: f64) -> Self {
        Self { w_0, g: 1.0, r: 0.5 }
    }

    fn get_mag_at(&self, w: f64) -> f64 {
        let g = self.g;
        let r = self.r;
        let w_0 = self.w_0;

        g * r * w_0.sin() / (((2.0 * w).cos() - 2.0 * r * w.cos() * w_0.cos() + r.powi(2)).powi(2) + ((2.0 * w).sin() - 2.0 * r * w.sin() * w_0.cos()).powi(2)).sqrt()
    }
}

#[derive(Debug)]
pub struct ResEstimator {
    resonators: Vec<ResonatorParams>,
    x: Vec<f64>,
    y: Vec<f64>,
}

/// exposes relevant error states of levenberg_marquardt::TerminationReason
#[derive(Debug)]
pub enum SolveError {
    /// Maximum number of iterations was hit
    LostPatience(ResEstimator),
    Numerical(&'static str),
}

impl Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for SolveError {}

impl ResEstimator {
    #[inline]
    pub fn new(x: Vec<f64>, y: Vec<f64>, frequencies: Vec<f64>) -> Self {
        let mut resonators = Vec::with_capacity(frequencies.len());
        for w_0 in frequencies {
            resonators.push(ResonatorParams::init(w_0));
        }
        Self {
            resonators,
            x,
            y,
        }
    }

    #[inline]
    pub fn solve(self) -> Result<Self, SolveError> {
        let (result, report) = LevenbergMarquardt::new().minimize(self);
        match report.termination {
            TerminationReason::Numerical(s) => Err(SolveError::Numerical(s)),
            TerminationReason::LostPatience => {
                // println!("{:?}, {:?}", report, result);
                Err(SolveError::LostPatience(result))
            },
            TerminationReason::User(_) 
            | TerminationReason::NoImprovementPossible(_) 
            | TerminationReason::NoParameters 
            | TerminationReason::NoResiduals 
            | TerminationReason::WrongDimensions(_) => unreachable!(),
            _ => Ok(result)
        }
    }

    #[inline]
    fn d_r(resonator: &ResonatorParams, w: f64) -> f64 {
        let r = resonator.r;
        let w_0 = resonator.w_0;
        (2.0 * w).cos() - 2.0 * r * w.cos() * w_0.cos() + r.powi(2)
    }

    #[inline]
    fn d_l(resonator: &ResonatorParams, w: f64) -> f64 {
        let r = resonator.r;
        let w_0 = resonator.w_0;
        (2.0 * w).sin() - 2.0 * r * w.sin() * w_0.cos()
    }

    #[inline]
    fn f_i(resonator: &ResonatorParams, d_sqrt: f64) -> f64 {
        let r = resonator.r;
        let w_0 = resonator.w_0;
        let g = resonator.g;
        g * r * w_0.sin() / d_sqrt
    }

    #[inline]
    fn del_d_del_r(resonator: &ResonatorParams, w: f64, d_r: f64, d_l: f64) -> f64 {
        let r = resonator.r;
        let w_0 = resonator.w_0;
        2.0 * d_r * (-2.0 * w_0.cos() * w.cos() + 2.0 * r) + 2.0 * d_l * (-2.0 * w_0.cos() * w.sin())
    }

    #[inline]
    fn del_f_del_r(resonator: &ResonatorParams, d_sqrt: f64, d: f64, del_d_del_r: f64) -> f64 {
        let r = resonator.r;
        let w_0 = resonator.w_0;
        let g = resonator.g;
        (g * w_0.sin() * d_sqrt - 1.0 / (2.0 * d_sqrt) * del_d_del_r * g * r * w_0.sin()) / d
    }

    #[inline]
    fn del_f_del_g(resonator: &ResonatorParams, d_sqrt: f64) -> f64 {
        resonator.r * resonator.w_0.sin() / d_sqrt
    }

    #[inline]
    fn get_residuals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len());
        for (x, y) in self.x.iter().zip(&self.y) {
            let infos = self.get_res_estimator_infos(*x, *y);
            let mut sum = 0.0;
            for info in infos {
                sum += info.f;
                //sum += BARRIER_SCALE_FACTOR * ((1.0 - info.resonator.r).ln() - info.resonator.r.ln());
            }
            out.push(sum - y);
        }
        out
    }

    #[inline]
    fn get_jacobian(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len() * 3);
        for (x, y) in self.x.iter().zip(&self.y) {
            let infos = self.get_res_estimator_infos(*x, *y);
            for info in infos {
                let derivs = info.get_derivs();
                out.push(derivs[0]);
                out.push(derivs[1]);
            }
        }
        out
    }

    #[inline]
    fn get_res_estimator_infos(&self, x: f64, y: f64) -> Vec<ResEstimatorInfo> {
        let mut infos = Vec::with_capacity(self.resonators.len());
        for resonator in &self.resonators {
            let d_r = Self::d_r(&resonator, x);
            let d_l = Self::d_l(&resonator, x);
            let d = d_r.powi(2) + d_l.powi(2);
            let f = Self::f_i(&resonator, d.sqrt());
            infos.push(ResEstimatorInfo { x, y, f, d, d_r, d_l, resonator });
        }
        infos
    }

    // fn get_jacobian(&self) -> Vec<f64> {
    //     let mut out = Vec::with_capacity(self.x.len() * 3);
    //     let (g, r, b) = (self.params.x, self.params.y, self.params.z);
    //     for (x, y) in self.x.iter().zip(&self.y) {
    //         out.push(Self::g_grad(g, r, self.w_0));
    //         out.push(Self::r_grad(g, r, *y, *x, self.w_0));
    //         out.push(Self::b_grad(g, r, b, *y, *x, self.w_0))
    //     }
    //     out
    // }
}

struct ResEstimatorInfo<'a> {
    x: f64,
    y: f64,
    f: f64,
    d: f64,
    d_r: f64,
    d_l: f64,
    resonator: &'a ResonatorParams,
}

impl<'a> ResEstimatorInfo<'a> {
    #[inline]
    pub fn get_derivs(&'a self) -> [f64; 2] {
        let del_d_del_r = ResEstimator::del_d_del_r(self.resonator, self.x, self.d_r, self.d_l);
        let del_f_del_r = ResEstimator::del_f_del_r(self.resonator, self.d, self.d.sqrt(), del_d_del_r);
        //let barriers = BARRIER_SCALE_FACTOR * (1.0 - self.resonator.r).recip() - self.resonator.r.recip();
        let del_f_del_g = ResEstimator::del_f_del_g(self.resonator, self.d.sqrt());
        [del_f_del_g, del_f_del_r]
    
    }
}

// m = number of residuals
// n = number of parameters
impl LeastSquaresProblem<f64, Dyn, Dyn> for ResEstimator {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, Dyn, Self::ParameterStorage>) {
        let mut iter = x.iter();
        for resonator in &mut self.resonators {
            resonator.g = *iter.next().unwrap();
            resonator.r = *iter.next().unwrap();
        }
    }

    fn params(&self) -> nalgebra::Vector<f64, Dyn, Self::ParameterStorage> {
        let mut out = Vec::with_capacity(self.resonators.len() * 2);
        for resonator in &self.resonators {
            out.push(resonator.g);
            out.push(resonator.r);
        }
        DVector::from(out)
    }

    fn residuals(&self) -> Option<nalgebra::Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(DVector::from(self.get_residuals()))
    }

    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        Some(DMatrix::from_row_slice(self.x.len(), self.resonators.len() * 2, &self.get_jacobian()))
    }
}

#[cfg(test)]
mod test {
    use crate::resonance_estimator;

    use super::*;
    use std::f64::consts::PI;
    use gp_resonator::resonator;
    use levenberg_marquardt::differentiate_numerically;
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn test_residual_calculation() {
        let w_0 = PI / 2.0;
        let r = 0.5;
        let g = 6.0;
        let xs = vec![PI / 4.0, PI / 2.0, 3.0 * PI / 4.0];

        for x in xs {
            let resonator_params = ResonatorParams { w_0, g, r };
            let y = resonator_params.get_mag_at(x);
            
            let default_resonator_params = ResonatorParams::init(w_0);
            let y_default = default_resonator_params.get_mag_at(x);
            let residual_expected = y_default - y;

            let res_estimator_test = ResEstimator::new(vec![x], vec![y], vec![w_0]);

            let residual = res_estimator_test.get_residuals()[0];
            assert_relative_eq!(residual, residual_expected, epsilon = 1e-13);
        }
        
    }

    #[test]
    fn test_basic() {
        let w_0 = PI / 2.0;
        let r = 0.25;
        let g = 6.0;
        const SIZE: usize = 1024;
        let mut x = Vec::with_capacity(SIZE);
        for i in 0..SIZE {
            x.push(i as f64 / SIZE as f64 * PI);
        }
        let resonator_params = ResonatorParams { w_0, g, r };
        let y = x.iter().map(|v| resonator_params.get_mag_at(*v)).collect::<Vec<f64>>();

        let mut res_estimator_test = ResEstimator::new(x, y, vec![w_0]);

        let jacobian_numerical = differentiate_numerically(&mut res_estimator_test).unwrap();
        let jacobian_trait = res_estimator_test.jacobian().unwrap();
        //assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13);

        let res_estimator = res_estimator_test.solve().unwrap();
        println!("{:?}", res_estimator.resonators);
    }

    #[test]
    fn test_more_complex() {
        let mut rng = rand::thread_rng();
        let mut resonators = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut params = ResonatorParams::init(rng.gen_range(0.0..PI));
            params.g = rng.gen_range(0.0..10.0);
            params.r = rng.gen_range(0.0000001..0.999999999);
            resonators.push(params);
        }
        println!("{:?}", resonators);

        const SIZE: usize = 64;
        let mut x = Vec::with_capacity(SIZE);
        let mut y = Vec::with_capacity(SIZE);
        for i in 0..SIZE {
            x.push(i as f64 / SIZE as f64 * PI);
            y.push(resonators.iter().map(|v| v.get_mag_at(i as f64 / SIZE as f64 * PI)).sum::<f64>());
        }

        let mut res_estimator_test = ResEstimator::new(x, y, resonators.iter().map(|v| v.w_0).collect::<Vec<f64>>());
        let resonance_estimator = res_estimator_test.solve().unwrap();
        println!("{:?}", resonance_estimator.resonators);
    }
}