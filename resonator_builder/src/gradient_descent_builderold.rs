use std::{fmt::{Debug, Display}, error::Error};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Vector3, Owned, U3, Dyn, DVector, MatrixXx3};
use num_complex::Complex;

const BARRIER_SCALE_FACTOR: f64 = 1e0;

#[derive(Debug)]
pub struct ResonatorParams {
    w_0: f64,
    g: f64,
    r: f64,
}

impl ResonatorParams {
    #[inline]
    pub fn init(w_0: f64) -> Self {
        Self { w_0, g: 1.0, r: 0.5 }
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
        assert!(x.len() == y.len());
        assert!(x.len() >= 3);

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


    // #[inline]
    // fn r_grad(g: f64, r: f64, y: f64, x: f64, w_0: f64) -> f64 {
    //     2.0 * g.powi(2) * r * w_0.sin().powi(2) - 4.0 * y.powi(2) * (r - w_0.cos() * x.cos()) * (r * (r - 2.0 * w_0.cos() * x.cos()) + (2.0 * x).cos()) + 8.0 * y.powi(2) * w_0.cos() * x.sin().powi(2) * (x.cos() - r * w_0.cos()) + BARRIER_SCALE_FACTOR * (1.0 / (1.0 - r) - 1.0 / r)
    // }

    // #[inline]
    // fn g_grad(g: f64, r: f64, w_0: f64) -> f64 {
    //     2.0 * g * r.powi(2) * w_0.sin().powi(2)
    // }

    // #[inline]
    // fn residual(g: f64, r: f64, y: f64, x: f64, w_0: f64) -> f64 {
    //     g.powi(2) * r.powi(2) * w_0.sin().powi(2) - y.powi(2) * (((2.0 * x).cos() - 2.0 * r * w_0.cos() * x.cos() + r.powi(2)).powi(2)+((2.0 * x).sin() - 2.0 * r * w_0.cos() * x.sin()).powi(2)) + BARRIER_SCALE_FACTOR * (-(1.0 - r).ln() - r.ln())
    // }

    #[inline]
    fn d_r(resonator: &ResonatorParams, w: f64) -> f64 {
        (2.0 * w).cos() - 2.0 * resonator.r * w.cos() * resonator.w_0.cos() + resonator.r.powi(2)
    }

    #[inline]
    fn d_l(resonator: &ResonatorParams, w: f64) -> f64 {
        (2.0 * w).sin() - 2.0 * resonator.r * w.sin() * resonator.w_0.cos()
    }

    #[inline]
    fn f_i(resonator: &ResonatorParams, d_sqrt: f64) -> f64 {
        resonator.g * resonator.r * resonator.w_0.sin() / d_sqrt
    }

    #[inline]
    fn del_d_del_r(resonator: &ResonatorParams, w: f64, d_r: f64, d_l: f64) -> f64 {
        2.0 * d_r * (2.0 * resonator.w_0.cos() * w.cos() + 2.0 * resonator.r) + 2.0 * d_l * (2.0 * resonator.w_0.cos() * w.sin())
    }

    #[inline]
    fn del_f_del_r(resonator: &ResonatorParams, d_sqrt: f64, d: f64, del_d_del_r: f64) -> f64 {
        (resonator.g * resonator.w_0.sin() * d_sqrt - 1.0 / (2.0 * d_sqrt) * del_d_del_r * resonator.g * resonator.r * resonator.w_0.sin()) / d
    }

    #[inline]
    fn del_f_del_g(resonator: &ResonatorParams, d_sqrt: f64) -> f64 {
        resonator.r * resonator.w_0.sin() / d_sqrt
    }

    #[inline]
    fn del_resi_del_r(sum: f64, del_f_del_r: f64, f: f64) -> f64 {
        sum * del_f_del_r + 2.0 * f * del_f_del_r
    }

    #[inline]
    fn del_resi_del_g(sum: f64, del_f_del_g: f64, f: f64) -> f64 {
        sum * del_f_del_g + 2.0 * f * del_f_del_g
    }

    #[inline]
    fn residual(y: f64, fs: Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for f in fs {
            sum += f;
        }
        sum * sum - 2.0 * y * sum + y * y
    }

    fn get_values(&self, x: f64, y: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let mut d_rs = Vec::with_capacity(self.resonators.len());
        let mut d_ls = Vec::with_capacity(self.resonators.len());
        for res in &self.resonators {
            d_rs.push(Self::d_r(res, x));
            d_ls.push(Self::d_l(res, x));
        }

        let mut fs = Vec::with_capacity(self.resonators.len());
        let mut del_f_del_rs = Vec::with_capacity(self.resonators.len());
        let mut del_f_del_gs = Vec::with_capacity(self.resonators.len());
        for (i, (d_r, d_l)) in d_rs.iter().zip(&d_ls).enumerate() {
            let d = d_r.powi(2) + d_l.powi(2);
            let d_sqrt = d.sqrt();
            fs.push(Self::f_i(&self.resonators[i], d_sqrt));
            let del_d_del_r = Self::del_d_del_r(&self.resonators[i], x, *d_r, *d_l);
            del_f_del_rs.push(Self::del_f_del_r(&self.resonators[i], d_sqrt, d, del_d_del_r));
            del_f_del_gs.push(Self::del_f_del_g(&self.resonators[i], d_sqrt));
        }

        let mut sums = Vec::with_capacity(self.resonators.len());
        for (c, f) in fs.iter().enumerate() {
            sums.push(0.0);
            for i in 0..self.resonators.len() {
                if i != c {
                    sums[c] += f;
                }
            }
            sums[c] -= 2.0 * y;
        }

        let mut del_resi_del_rs = Vec::with_capacity(self.resonators.len());
        let mut del_resi_del_gs = Vec::with_capacity(self.resonators.len());
        for (i, ((sum, del_f_del_r), f)) in sums.iter().zip(&del_f_del_rs).zip(&fs).enumerate() {
            del_resi_del_rs.push(Self::del_resi_del_r(*sum, *del_f_del_r, *f));
            del_resi_del_gs.push(Self::del_resi_del_g(*sum, del_f_del_gs[i], *f));
        }

        let residual = Self::residual(y, fs);

        (del_resi_del_rs, del_resi_del_gs, residual)
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

// m = number of residuals
// n = number of parameters
impl LeastSquaresProblem<f64, Dyn, Dyn> for ResEstimator {
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

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

}