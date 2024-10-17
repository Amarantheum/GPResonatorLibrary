use num_complex::Complex;
use nalgebra::DMatrix;

use crate::plan_types::PhasedResonatorArrayPlan;

pub struct ARRational {
    poles: Vec<Complex<f64>>,
    partial_fraction_coefficients: Vec<Complex<f64>>,
}

impl ARRational {
    pub fn new(mut coefficients: Vec<f64>, poles: Vec<Complex<f64>>) -> Self {
        let mut first_non_zero = coefficients.len();
        for f in coefficients.iter().rev() {
            if *f != 0.0 {
                break;
            }
            first_non_zero -= 1;
        }
        coefficients.truncate(first_non_zero);
        assert!(coefficients.len() > 1);
        assert!(poles.len() == coefficients.len() - 1);
        let mut complex_coefficients = Vec::with_capacity(coefficients.len());
        for c in coefficients {
            complex_coefficients.push(Complex::new(c, 0.0));
        }

        let partial_fraction_coefficients = Self::get_partial_fraction_coefficients(&complex_coefficients, &poles);

        Self {
            poles,
            partial_fraction_coefficients,
        }
    }

    fn get_partial_fraction_coefficients(coefficients: &Vec<Complex<f64>>, poles: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        let coefficient_matrix = Self::get_coefficient_matrix(coefficients, poles);
        // solve Ax = b
        // where A is the coefficient matrix (each factor is a column) and b is [0 0 ... 1]
        let coefficient_matrix = DMatrix::from_fn(coefficient_matrix.len(), coefficient_matrix[0].len(), |i, j| coefficient_matrix[j][i]);
        let b = DMatrix::from_fn(coefficient_matrix.nrows(), 1, |i, _| if i == poles.len() - 1 { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) });

        let x = coefficient_matrix.lu().solve(&b).unwrap();
        let mut result = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            result.push(x[(i, 0)]);
        }
        result
    }


    fn get_coefficient_matrix(coefficients: &Vec<Complex<f64>>, poles: &Vec<Complex<f64>>) -> Vec<Vec<Complex<f64>>> {
        let mut result = Vec::with_capacity(poles.len());
        for i in 0..poles.len() {
            result.push(Self::get_factor(coefficients, poles, i));
        }
        result
    }

    fn get_factor(coefficients: &Vec<Complex<f64>>, poles: &Vec<Complex<f64>>, index: usize) -> Vec<Complex<f64>> {
        // solve the problem (1 + a_1 * z^-1 + a_2 * z^-2 + ... + a_n * z^-n) / (1 - poles[index] * z^-1)
        let divisor_coef = -poles[index];

        let mut result = Vec::with_capacity(coefficients.len() - 1);

        let mut current_coef = coefficients[coefficients.len() - 1];
        for i in 0..coefficients.len() - 1 {
            let new_coef = current_coef / divisor_coef;
            result.push(new_coef);
            current_coef = coefficients[coefficients.len() - 2 - i] - new_coef
        }
        debug_assert!(current_coef.norm() < 1e-10, "One of the given poles is not a pole of the rational function: function: {:?}, pole: {}", coefficients, divisor_coef);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn test_get_factor() {
        let coefficients = vec![1.0, 1.0, 1.0];
        let coefficients = coefficients.iter().map(|c| Complex::new(*c, 0.0)).collect();
        let poles = vec![Complex::new(2.0, 0.0) / Complex::new(-1.0, 3_f64.sqrt()), Complex::new(2.0, 0.0) / Complex::new(-1.0, -3_f64.sqrt())];
        let factors = ARRational::get_factor(&coefficients, &poles, 0);

        let expected_0 = -Complex::new(2.0, 0.0) / Complex::new(-1.0, -3_f64.sqrt());
        assert_float_eq!(factors[0].re, expected_0.re, abs <= 1e-10);
        assert_float_eq!(factors[0].im, expected_0.im, abs <= 1e-10);
        assert_float_eq!(factors[1].re, 1.0, abs <= 1e-10);
        assert_float_eq!(factors[1].im, 0.0, abs <= 1e-10);

        let factors = ARRational::get_factor(&coefficients, &poles, 1);
        let expected_0 = -Complex::new(2.0, 0.0) / Complex::new(-1.0, 3_f64.sqrt());
        assert_float_eq!(factors[0].re, expected_0.re, abs <= 1e-10);
        assert_float_eq!(factors[0].im, expected_0.im, abs <= 1e-10);
        assert_float_eq!(factors[1].re, 1.0, abs <= 1e-10);
        assert_float_eq!(factors[1].im, 0.0, abs <= 1e-10);
    }

    #[test]
    fn test_get_partial_coefficients() {
        let coefficients = vec![1.0, 1.0, 1.0];
        let coefficients = coefficients.iter().map(|c| Complex::new(*c, 0.0)).collect();
        let poles = vec![Complex::new(2.0, 0.0) / Complex::new(-1.0, 3_f64.sqrt()), Complex::new(2.0, 0.0) / Complex::new(-1.0, -3_f64.sqrt())];
        let factors = ARRational::get_partial_fraction_coefficients(&coefficients, &poles);

        let expcted1 = Complex::new(2.0, 0.0) / Complex::new(3.0, 3_f64.sqrt());
        let expected2 = Complex::new(2.0, 0.0) / Complex::new(3.0, -3_f64.sqrt());
        assert_float_eq!(factors[0].re, expcted1.re, abs <= 1e-10);
        assert_float_eq!(factors[0].im, expcted1.im, abs <= 1e-10);
        assert_float_eq!(factors[1].re, expected2.re, abs <= 1e-10);
        assert_float_eq!(factors[1].im, expected2.im, abs <= 1e-10);
    }

    #[test]
    fn test_get_partial_coefficients_real() {
        let coefficients = vec![1.0, 0.0, -1.0];
        let coefficients = coefficients.iter().map(|c| Complex::new(*c, 0.0)).collect();
        let poles = vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
        let factors = ARRational::get_partial_fraction_coefficients(&coefficients, &poles);
        println!("{:?}", factors);
        println!("{:?}", Complex::new(2.0, 0.0) / Complex::new(3.0, -3_f64.sqrt()));
        println!("{:?}", Complex::new(2.0, 0.0) / Complex::new(3.0, 3_f64.sqrt()));
    }

    #[test]
    fn test_get_partial_coefficients_real_2() {
        let coefficients = vec![1.0, -1.5, 0.5];
        let coefficients = coefficients.iter().map(|c| Complex::new(*c, 0.0)).collect();
        let poles = vec![Complex::new(0.5, 0.0), Complex::new(1.0, 0.0)];
        let factors = ARRational::get_partial_fraction_coefficients(&coefficients, &poles);
        println!("{:?}", factors);
        println!("{:?}", Complex::new(2.0, 0.0) / Complex::new(3.0, -3_f64.sqrt()));
        println!("{:?}", Complex::new(2.0, 0.0) / Complex::new(3.0, 3_f64.sqrt()));
    }
}