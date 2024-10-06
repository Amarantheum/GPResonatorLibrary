use num_complex::Complex;
use nalgebra::DMatrix;

pub struct ARRational {
    coefficients: Vec<Complex<f64>>,
    poles: Vec<Complex<f64>>,
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
        Self {
            coefficients: complex_coefficients,
            poles,
        }
    }

    // pub fn get_coefficent_matrix(&self) -> Vec<Vec<f64>> {

    // }

    fn get_factor(&self, index: usize) -> Vec<Complex<f64>> {
        // solve the problem (1 + a_1 * z^-1 + a_2 * z^-2 + ... + a_n * z^-n) / (1 - p_index * z^-1)
        let divisor_coef = -self.poles[index];

        let mut result = Vec::with_capacity(self.coefficients.len() - 1);

        let mut current_coef = self.coefficients[self.coefficients.len() - 1];
        for i in 0..self.coefficients.len() - 1 {
            let new_coef = current_coef / divisor_coef;
            result.push(new_coef);
            current_coef = self.coefficients[self.coefficients.len() - 2 - i] - new_coef
        }
        debug_assert!(current_coef.norm() < 1e-10);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_factor() {
        let coefficients = vec![1.0, 1.0, 1.0];
        let poles = vec![Complex::new(2.0, 0.0) / Complex::new(-1.0, 3_f64.sqrt()), Complex::new(2.0, 0.0) / Complex::new(-1.0, -3_f64.sqrt())];
        let ar = ARRational::new(coefficients, poles);
        let factors = ar.get_factor(0);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], Complex::new(-0.5, 0.0));
        assert_eq!(factors[1], Complex::new(-1.0, 0.0));
    }
}