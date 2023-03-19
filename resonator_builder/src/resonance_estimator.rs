use std::{fmt::{Debug, Display}, error::Error};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Vector3, Owned, U3, Dyn, DVector, MatrixXx3};
use num_complex::Complex;

pub struct ResEstimator {
    w_0: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    params: Vector3<f64>,
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

impl Debug for ResEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResEstimator")
            .field("g", &self.params.x)
            .field("r", &self.params.y)
            .field("b", &self.params.z)
            .field("w_0", &self.w_0)
            .finish()
    }
}

impl ResEstimator {
    #[inline]
    pub fn new(x: Vec<f64>, y: Vec<f64>, w_0: f64) -> Self {
        assert!(x.len() == y.len());
        assert!(x.len() >= 3);
        Self {
            w_0,
            x,
            y,
            params: Vector3::new(1.0, 0.5, 0.0),
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
    pub fn into_res_params(&self) -> ResParams {
        ResParams { g: self.params.x.abs(), r: self.params.y.min(2.0 - self.params.y), b: self.params.z, w_0: self.w_0 }
    }

    #[inline]
    fn r_grad(g: f64, r: f64, b: f64, y: f64, x: f64, w_0: f64) -> f64 {
        2.0 * g.powi(2) * r * w_0.sin().powi(2) - 4.0 * (y - b).powi(2) * (r - w_0.cos() * x.cos()) * (r * (r - 2.0 * w_0.cos() * x.cos()) + (2.0 * x).cos()) + 8.0 * (y - b).powi(2) * w_0.cos() * x.sin().powi(2) * (x.cos() - r * w_0.cos())
    }

    #[inline]
    fn g_grad(g: f64, r: f64, w_0: f64) -> f64 {
        2.0 * g * r.powi(2) * w_0.sin().powi(2)
    }

    #[inline]
    fn b_grad(g: f64, r: f64, b: f64, y: f64, x: f64, w_0: f64) -> f64 {
        2.0 * (y - b) * (((2.0 * x).cos() + r * (r - 2.0 * x.cos() * w_0.cos())).powi(2) + ((2.0 * x).sin() - 2.0 * r * w_0.cos() * x.sin()).powi(2))
    }

    #[inline]
    fn residual(g: f64, r: f64, b: f64, y: f64, x: f64, w_0: f64) -> f64 {
        g.powi(2) * r.powi(2) * w_0.sin().powi(2) - (y - b).powi(2) * (((2.0 * x).cos() - 2.0 * r * w_0.cos() * x.cos() + r.powi(2)).powi(2)+((2.0 * x).sin() - 2.0 * r * w_0.cos() * x.sin()).powi(2))
    }

    fn get_residuals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len());
        let (g, r, b) = (self.params.x, self.params.y, self.params.z);
        for (x, y) in self.x.iter().zip(&self.y) {
            out.push(Self::residual(g, r, b, *y, *x, self.w_0));
        }
        out
    }

    fn get_jacobian(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len() * 3);
        let (g, r, b) = (self.params.x, self.params.y, self.params.z);
        for (x, y) in self.x.iter().zip(&self.y) {
            out.push(Self::g_grad(g, r, self.w_0));
            out.push(Self::r_grad(g, r, b, *y, *x, self.w_0));
            out.push(Self::b_grad(g, r, b, *y, *x, self.w_0))
        }
        out
    }
}

// m = number of residuals
// n = number of parameters
impl LeastSquaresProblem<f64, Dyn, U3> for ResEstimator {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U3>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, U3, Self::ParameterStorage>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> nalgebra::Vector<f64, U3, Self::ParameterStorage> {
        self.params
    }

    fn residuals(&self) -> Option<nalgebra::Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(DVector::from(self.get_residuals()))
    }

    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, Dyn, U3, Self::JacobianStorage>> {
        Some(MatrixXx3::from_row_slice(&self.get_jacobian()))
    }
}

/// A data structure that stores the parameters that [`ResEstimator`] solves for and the center frequency of the resonance.
pub struct ResParams {
    /// the gain of the resonance
    pub g: f64,
    /// the radius of the conjugate poles
    pub r: f64,
    /// offset in the y axis (this is primarily used to make estimation for multiple resonances more accurate)
    pub b: f64,
    /// Center frequency of the resonance
    pub w_0: f64,
}

impl ResParams {
    /// This function returns the value of:
    /// let z = cos(w) + j * sin(w)
    /// |rsin(w_0)z/[(z-re^(jw_0))*(z-re^(-jw_0))]|
    #[inline]
    pub fn predict_r_influence(&self, w: f64) -> f64 {
        let z = Complex::new(w.cos(), w.sin());
        let r = self.r;
        let pole = Complex::new(self.w_0.cos(), self.w_0.sin());
        (Complex::new(r*self.w_0.sin(), 0.0) * z / (z - pole.scale(r)) / (z - pole.conj().scale(r))).norm()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_find_peaks() {
        let x = vec![0.6853981633974483, 0.6953981633974483, 0.7053981633974483, 0.7153981633974482, 0.7253981633974482, 0.7353981633974482, 0.7453981633974482, 0.7553981633974483, 0.7653981633974483, 0.7753981633974483, 0.7853981633974483, 0.7953981633974483, 0.8053981633974483, 0.8153981633974483, 0.8253981633974483, 0.8353981633974483, 0.8453981633974483, 0.8553981633974483, 0.8653981633974482, 0.8753981633974482];
        let y = vec![7.463064026124787, 8.246068583357303, 9.225654806192592, 10.486023747215427, 12.167480639272394, 14.522491287795539, 18.055707274533074, 23.943418624714, 35.70594745948763, 70.78491631036313, 707.4604229211972, 70.08058329541748, 34.99887600954063, 23.23567942705814, 17.34754809801119, 13.813933403148289, 11.458489189592225, 9.776544487140963, 8.515625237913223, 7.535422107988907];

        let mut pp = ResEstimator::new(x, y, PI / 4.0);
        let (result, report) = LevenbergMarquardt::new().minimize(pp);
        println!("{:?}", report);
        println!("{:?}", result);
    }

    #[test]
    fn test_find_peaks2() {
        let x = vec![0.3757294192327392, 0.3757773561323607, 0.3758252930319821, 0.3758732299316035, 0.37592116683122495, 0.37596910373084635, 0.3760170406304678, 0.3760649775300892];
        let y = vec![0.19417270978172357, 0.39168883476639427, 0.7549296185189659, 0.9943495985986138, 0.9552361536329542, 0.6490050104982352, 0.2766297165111483, 0.19921037252873267];

        let mut pp = ResEstimator::new(x, y, 0.3758732299316035,);
        let (result, report) = LevenbergMarquardt::new().minimize(pp);
        println!("{:?}", report);
        println!("{:?}", result);
    }
}