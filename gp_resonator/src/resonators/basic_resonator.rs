use std::f64::consts::PI;
use bode_plot::plot::{BodePlotTransferFunction, LTISystem};
use num_complex::Complex;

/// Represents a filter with 2 conjugate poles on the Z-plane with a gain amount.
/// This results in frequencies closer to the argument of the poles being amplified.
#[derive(Debug, Clone)]
pub struct ConjPoleResonator {
    /// The real part of the pole location doubled
    re_2: f64,
    /// The magnitude of the pole location squared
    mag_sq: f64,
    /// The gain on the resonance
    gain: f64,

    /// The magnitude of the pole location (stored for updating resonator)
    mag: f64,
    /// The angle of the pole location (stored forr updating resonator)
    arg: f64,
}

impl ConjPoleResonator {
    /// Create a new [`ConjPoleResonator`] given a complex number in polar form.
    /// The poles will be at the location of the complex number and its conjugate.
    /// # Arguments
    /// * `mag` - The magnitude of the complex number (must be in range [0, 1))
    /// * `arg` - The argument of the commplex number (must be in range [0, π])
    /// * `gain` - The gain of the filter
    #[inline]
    pub fn new_polar(mag: f64, arg: f64, gain: f64) -> Self {
        debug_assert!(mag < 1.0 && mag >= 0.0);
        debug_assert!(arg >= 0.0 && arg <= PI);
        Self {
            re_2: mag * arg.cos() * 2.0,
            mag_sq: mag * mag,
            gain,
            arg,
            mag,
        }
    }

    /// Set the magnitude of the pole of the resonator
    /// # Arguments
    /// * `mag` - The magnitude of the complex number (must be in range [0, 1))
    #[inline]
    pub fn set_mag(&mut self, mag: f64) {
        debug_assert!(mag < 1.0 && mag >= 0.0);
        self.re_2 = mag * self.arg.cos() * 2.0;
        self.mag_sq = mag * mag;
        self.mag = mag;
    }

    /// Set the angle of the pole of the resonator
    /// # Arguments
    /// `arg` - The argument of the commplex number (must be in range [0, π])
    #[inline]
    pub fn set_arg(&mut self, arg: f64) {
        debug_assert!(arg >= 0.0 && arg <= PI);
        self.re_2 = self.mag * arg.cos() * 2.0;
        self.arg = arg;
    }

    #[inline]
    fn process_vec(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut buf = vec![0.0; x.len()];
        if x.len() > 1 {
            buf[1] = self.gain * x[0];
            self.process_buf(&x[..], &mut buf[..]);
        }
        return buf
    }

    #[inline]
    pub fn process_buf(&self, x: &[f64], buf: &mut [f64]) {
        debug_assert!(x.len() == buf.len());
        for i in 0..buf.len() {
            buf[i] = x[i-1] * self.gain + self.re_2 * buf[i - 1] - self.mag_sq * buf[i - 2];
        }
    }

    /// Process a single data point given the required values from the difference equation
    /// and returns `y[n]`. Prefer [`Self::process_buf`] or [`Self::process_buf_add`] for better cache performance.
    /// # Arguments
    /// * `x_1` - value of `x[n-1]`
    /// * `y_1` - value of `y[n-1]`
    /// * `y_2` - value of `y[n-2]`
    #[inline]
    pub fn process_single(&self, x_1: f64, y_1: f64, y_2: f64) -> f64 {
        x_1 * self.gain + self.re_2 * y_1 - self.mag_sq * y_2
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

impl LTISystem for ConjPoleResonator {
    fn process(&self, samples: &Vec<f64>) -> Vec<f64> {
        self.process_vec(samples)
    }
}

#[cfg(test)]
mod tests {
    use std::{time::Instant, f64::consts::PI};

    use bode_plot::{create_plot, DEFAULT_WIDTH, DEFAULT_HEIGHT};
    use wav_util::*;

    use super::*;

    #[test]
    fn test_resonator_sections() {
        let mut array = ConjPoleResonator::new_polar(0.99, PI / 32.0, 1.0);
        let ([chan1, chan2], _) = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = Instant::now();
        let middle = chan1.len() / 2;
        array.process_buf(&chan1[..middle], &mut out_chan1[..middle]);
        array.process_buf(&chan1[middle..], &mut out_chan1[middle..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_harmonic_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}