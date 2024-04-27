use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Owned, Dyn, DVector, DMatrix};

use std::error::Error;
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, Copy)]
struct ApproxResParams {
    pub w_0: f64,
    pub g: f64,
    pub r: f64,
}

impl ApproxResParams {
    #[inline]
    pub fn init(w_0: f64) -> Self {
        Self { w_0, g: 1.0, r: 0.9999999 }
    }

    #[inline]
    pub fn new(w_0: f64, g: f64, r: f64) -> Self {
        Self { w_0, g, r }
    }

    // pub fn get_log_mag_at(&self, w: f64) -> f64 {
    //     let g = self.g;
    //     let r = self.r;
    //     let w_0 = self.w_0;

    //     g.ln() + r.ln() + ((w - w_0) * (w - w_0) + 1.0).ln() - ((w - w_0) * (w - w_0) + (1.0 - r) * (1.0 - r)).ln()
    // }

    pub fn get_mag_at(&self, w: f64) -> f64 {
        let g = self.g;
        let r = self.r;
        let w_0 = self.w_0;

        g * r * ((w - w_0).powi(2) + 1.0) / ((w - w_0).powi(2) + (1.0 - r).powi(2))
    }
}

/// exposes relevant error states of levenberg_marquardt::TerminationReason
#[derive(Debug)]
pub enum SolveError {
    /// Maximum number of iterations was hit
    LostPatience(LogGradientDescentEstimator),
    Numerical(&'static str),
}

impl Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for SolveError {}

#[derive(Debug)]
pub struct LogGradientDescentEstimator {
    x: Vec<f64>,
    y: Vec<f64>,
    resonators: Vec<ApproxResParams>,
}

impl LogGradientDescentEstimator {
    #[inline]
    pub fn new(x: Vec<f64>, y: Vec<f64>, frequencies: Vec<f64>) -> Self {
        assert!(x.len() == y.len());
        let resonators = frequencies.into_iter().map(|v| ApproxResParams::init(v)).collect();
        Self {
            x,
            y,
            resonators,
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
    fn r_grad(g: f64, r: f64, x: f64, w_0: f64) -> f64 {
        g * ((x - w_0).powi(2) + 1.0) * ((x - w_0).powi(2) + 1.0 - r.powi(2)) / ((x - w_0).powi(2) + (1.0 - r).powi(2)).powi(2)
    }

    #[inline]
    fn g_grad(g: f64, r: f64, x: f64, w_0: f64) -> f64 {
        r * ((x - w_0).powi(2) + 1.0) / ((x - w_0).powi(2) + (1.0 - r).powi(2))
    }

    #[inline]
    fn get_residuals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len());
        for (x, y) in self.x.iter().zip(&self.y) {
            let mut sum = 0.0;
            for resonator in &self.resonators {
                sum += resonator.get_mag_at(*x);
            }
            out.push(sum - y);
        }
        out
    }

    #[inline]
    fn get_jacobian(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len() * self.resonators.len() * 2);
        for x in self.x.iter() {
            for resonator in &self.resonators {
                out.push(Self::g_grad(resonator.g, resonator.r, *x, resonator.w_0));
                out.push(Self::r_grad(resonator.g, resonator.r, *x, resonator.w_0));
            }
        }
        out
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LogGradientDescentEstimator {
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
    use plotters::prelude::*;

    fn plot_x_y(x: &Vec<f64>, y: &Vec<f64>, name: &str) {
        let y_max = y.iter().fold(0.0, |acc: f64, &v| acc.max(v));

        let root = BitMapBackend::new(name, (1920, 1080)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Gaussian Estimation", ("sans-serif", 30).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..PI, -5.0..y_max)
            .unwrap();
        chart.configure_mesh().draw().unwrap();
        chart.draw_series(LineSeries::new(x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)), &RED)).unwrap();
    }

    #[test]
    fn test_basic() {
        let w_0 = PI / 2.0;
        let r = 0.999;
        let g = 6.0;
        const SIZE: usize = 1024;
        let mut x = Vec::with_capacity(SIZE);
        for i in 0..SIZE {
            x.push(i as f64 / SIZE as f64 * PI);
        }
        let resonator_params = ApproxResParams { w_0, g, r };
        let y = x.iter().map(|v| resonator_params.get_mag_at(*v)).collect::<Vec<f64>>();
        plot_x_y(&x, &y, "approx_y.png");

        let mut res_estimator_test = LogGradientDescentEstimator::new(x.clone(), y, vec![w_0]);

        //let jacobian_numerical = differentiate_numerically(&mut res_estimator_test).unwrap();
        //let jacobian_trait = res_estimator_test.jacobian().unwrap();
        //assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13);

        let res_estimator = res_estimator_test.solve().unwrap();
        println!("{:?}", res_estimator.resonators);
        let y_est = x.iter().map(|v| res_estimator.resonators[0].get_mag_at(*v)).collect::<Vec<f64>>();
        plot_x_y(&x, &y_est, "approx_y_est.png");
    }

    #[test]
    fn test_approx_more_complex() {
        let mut rng = rand::thread_rng();
        let mut resonators = Vec::with_capacity(4);
        for _ in 0..5 {
            let mut params = ApproxResParams::init(rng.gen_range(0.0..PI));
            params.g = rng.gen_range(0.0..10.0);
            let exp = rng.gen_range(2..4);
            params.r = 1.0 - 1.0 / 10.0_f64.powi(exp);
            resonators.push(params);
        }
        println!("{:?}", resonators);

        const SIZE: usize = 1024;
        let mut x = Vec::with_capacity(SIZE);
        let mut y = Vec::with_capacity(SIZE);
        for i in 0..SIZE {
            x.push(i as f64 / SIZE as f64 * PI);
            y.push(resonators.iter().map(|v| v.get_mag_at(i as f64 / SIZE as f64 * PI)).sum::<f64>());
        }

        plot_x_y(&x, &y, "approx_y_complex.png");

        let mut res_estimator_test = LogGradientDescentEstimator::new(x.clone(), y, resonators.iter().map(|v| v.w_0).collect::<Vec<f64>>());
        let resonance_estimator = match res_estimator_test.solve() {
            Ok(v) => v,
            Err(e) => {
                match e {
                    SolveError::LostPatience(v) => v,
                    _ => panic!("Unexpected error: {:?}", e),
                }
            }
        };
        
        let y_ln_est = x.iter().map(|v| resonance_estimator.resonators.iter().map(|r| r.get_mag_at(*v)).sum::<f64>()).collect::<Vec<f64>>();
        plot_x_y(&x, &y_ln_est, "approx_y_est_complex.png");
        println!("{:?}", resonance_estimator.resonators);
    }
}