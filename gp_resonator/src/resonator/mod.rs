//! This module contains useful abstractions for resonating filters.
//! You are probably looking for [`ConjPoleResonator`].

use bode_plot::plot::BodePlotTransferFunction;
use num_complex::Complex;

/// Represents a filter with 2 conjugate poles on the Z-plane with a gain amount
pub struct ConjPoleResonator {
    /// The real part of the pole location doubled
    re_2: f64,
    /// The magnitude of the pole location squared
    mag_sq: f64,
    /// The gain on the resonance
    gain: f64,
}

impl ConjPoleResonator {
    /// Create a new [`ConjPoleResonator`] given a complex number in polar form.
    /// The poles will be at the location of the complex number and its conjugate.
    /// # Arguments
    /// * `mag` - The magnitude of the complex number
    /// * `arg` - The argument of the commplex number
    /// * `gain` - The gain of the filter
    #[inline]
    pub fn new_polar(mag: f64, arg: f64, gain: f64) -> Self {
        Self {
            re_2: mag * arg.cos() * 2.0,
            mag_sq: mag * mag,
            gain,
        }
    }

    // The difference equation for the filter
    #[inline]
    fn process(&self, x: &Vec<f64>, y: &mut Vec<f64>, n: usize) {
        y[n] = self.gain * x[n] - self.re_2 * y[n - 1] - self.mag_sq * y[n - 2];
    }

    /// Returns the locations of the two poles in the filter
    /// The first returned pole will have a positive imaginary component.
    #[inline]
    pub fn get_pole_locs(&self) -> (Complex<f64>, Complex<f64>) {
        let theta = (self.re_2 / 2.0 / self.mag_sq.sqrt()).acos();
        (Complex::from_polar(self.mag_sq.sqrt(), theta), Complex::from_polar(self.mag_sq.sqrt(), -theta))
    }
}

impl BodePlotTransferFunction for ConjPoleResonator {
    fn get_value(&self, z: Complex<f64>) -> Complex<f64> {
        let poles = self.get_pole_locs();
        ((z - poles.0) * (z - poles.1)).inv().scale(self.gain)
    }
}