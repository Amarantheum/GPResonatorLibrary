use std::fmt::Debug;

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Vector2, Owned, U2, Dyn, DVector, MatrixXx2};

struct ResEstimator {
    w_0: f64,
    x: Vec<f64>,
    y_sq: Vec<f64>,
    params: Vector2<f64>,
}

impl Debug for ResEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResEstimator")
            .field("g", &self.params.x)
            .field("r", &self.params.y)
            .finish()
    }
}

impl ResEstimator {
    pub fn new(x: Vec<f64>, y: Vec<f64>, w_0: f64) -> Self {
        Self {
            w_0,
            x,
            y_sq: y.into_iter().map(|v| v * v).collect(),
            params: Vector2::new(1.0, 0.5),
        }
    }

    #[inline]
    fn r_grad(g: f64, r: f64, y_sq: f64, x: f64, w_0: f64) -> f64 {
        2.0 * g.powi(2) * r * w_0.sin().powi(2) - 4.0 * y_sq * (r - w_0.cos() * x.cos()) * (r * (r - 2.0 * w_0.cos() * x.cos()) + (2.0 * x).cos()) + 8.0 * y_sq * w_0.cos() * x.sin().powi(2) * (x.cos() - r * w_0.cos())
    }

    #[inline]
    fn g_grad(g: f64, r: f64, w_0: f64) -> f64 {
        2.0 * g * r.powi(2) * w_0.sin().powi(2)
    }

    #[inline]
    fn residual(g: f64, r: f64, y_sq: f64, x: f64, w_0: f64) -> f64 {
        g.powi(2) * r.powi(2) * w_0.sin().powi(2) - y_sq * (((2.0 * x).cos() - 2.0 * r * w_0.cos() * x.cos() + r.powi(2)).powi(2)+((2.0 * x).sin() - 2.0 * r * w_0.cos() * x.sin()).powi(2))
    }

    fn get_residuals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len());
        let (g, r) = (self.params.x, self.params.y);
        for (x, y_sq) in self.x.iter().zip(&self.y_sq) {
            out.push(Self::residual(g, r, *y_sq, *x, self.w_0));
        }
        out
    }

    fn get_jacobian(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.x.len() * 2);
        let (g, r) = (self.params.x, self.params.y);
        for (x, y_sq) in self.x.iter().zip(&self.y_sq) {
            out.push(Self::g_grad(g, r, self.w_0));
            out.push(Self::r_grad(g, r, *y_sq, *x, self.w_0));
        }
        out
    }
}

// m = number of residuals
// n = number of parameters
impl LeastSquaresProblem<f64, Dyn, U2> for ResEstimator {
    type ParameterStorage = Owned<f64, U2>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U2>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, U2, Self::ParameterStorage>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> nalgebra::Vector<f64, U2, Self::ParameterStorage> {
        self.params
    }

    fn residuals(&self) -> Option<nalgebra::Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(DVector::from(self.get_residuals()))
    }

    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(MatrixXx2::from_row_slice(&self.get_jacobian()))
    }
}