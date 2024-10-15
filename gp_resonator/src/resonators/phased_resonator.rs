//! This module contains useful abstractions for resonating filters.
//! You are probably looking for [`PhasedConjPoleResonator`].

use std::f64::consts::PI;

use bode_plot::plot::{BodePlotTransferFunction, LTISystem};
use num_complex::Complex;

/// Represents a filter with 2 conjugate poles on the Z-plane with a gain amount.
/// This results in frequencies closer to the argument of the poles being amplified.
#[derive(Debug, Clone)]
pub struct PhasedConjPoleResonator {
    /// The real part of the pole location doubled
    re_2: f64,
    /// The magnitude of the pole location squared
    mag_sq: f64,

    /// The magnitude of the pole location (stored for updating resonator)
    mag: f64,
    /// The angle of the pole location (stored forr updating resonator)
    arg: f64,

    /// The phase of the resonator
    phase: f64,
    /// The constant term in the difference equation for x[n]
    /// 
    /// This is determined by `gain * cos(phase)`
    x_0_const: f64,
    /// The constant term in the difference equation for x[n-1]
    /// 
    /// This is determined by `gain * mag * (sin(phase) * sin(arg) - cos(phase) * cos(arg))`
    x_1_const: f64,
    /// The gain on the resonance
    gain: f64,
}

impl PhasedConjPoleResonator {
    /// Create a new [`PhasedConjPoleResonator`] given a complex number in polar form.
    /// The poles will be at the location of the complex number and its conjugate.
    /// # Arguments
    /// * `mag` - The magnitude of the complex number (must be in range [0, 1))
    /// * `arg` - The argument of the commplex number (must be in range [0, π])
    /// * `gain` - The gain of the filter
    #[inline]
    pub fn new_polar(mag: f64, arg: f64, phase: f64, gain: f64) -> Self {
        debug_assert!(mag < 1.0 && mag >= 0.0);
        debug_assert!(arg >= 0.0 && arg <= PI);
        debug_assert!(phase >= 0.0 && phase <= 2.0 * PI);
        Self {
            re_2: mag * arg.cos() * 2.0,
            mag_sq: mag * mag,
            arg,
            mag,
            phase,
            x_0_const: gain * phase.cos(),
            x_1_const: gain * mag * (phase.sin() * arg.sin() - phase.cos() * arg.cos()),
            gain,
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
        self.x_1_const = self.gain * mag * (self.phase.sin() * self.arg.sin() - self.phase.cos() * self.arg.cos());
        self.mag = mag;
    }

    /// Set the angle of the pole of the resonator
    /// # Arguments
    /// `arg` - The argument of the commplex number (must be in range [0, π])
    #[inline]
    pub fn set_arg(&mut self, arg: f64) {
        debug_assert!(arg >= 0.0 && arg <= PI);
        self.re_2 = self.mag * arg.cos() * 2.0;
        self.x_1_const = self.gain * self.mag * (self.phase.sin() * arg.sin() - self.phase.cos() * arg.cos());
        self.arg = arg;
    }

    #[inline]
    pub fn set_gain(&mut self, gain: f64) {
        self.x_0_const = gain * self.phase.cos();
        self.x_1_const = gain * self.mag * (self.phase.sin() * self.arg.sin() - self.phase.cos() * self.arg.cos());
        self.gain = gain;
    }

    #[inline]
    pub fn set_phase(&mut self, phase: f64) {
        self.x_0_const = self.gain * phase.cos();
        self.x_1_const = self.gain * self.mag * (phase.sin() * self.arg.sin() - phase.cos() * self.arg.cos());
        self.phase = phase;
    }

    pub fn process_buf(&self, x: &[f64], buf: &mut [f64]) {
        debug_assert!(x.len() == buf.len());
        if x.len() == 0 {
            return;
        }
        buf[0] = self.x_0_const * x[0];
        if x.len() == 1 {
            return;
        }
        buf[1] = self.x_0_const * x[1] + self.x_1_const * x[0] + self.re_2 * buf[0];
        for i in 2..x.len() {
            buf[i] = self.x_0_const * x[i] + self.x_1_const * x[i - 1] + self.re_2 * buf[i - 1] - self.mag_sq * buf[i - 2];
        }
    }

    /// Process a single data point given the required values from the difference equation
    /// and returns `y[n]`. Prefer [`Self::process_buf`] or [`Self::process_buf_add`] for better cache performance.
    /// # Arguments
    /// * `x_0` - value of `x[n]`
    /// * `x_1` - value of `x[n-1]`
    /// * `y_1` - value of `y[n-1]`
    /// * `y_2` - value of `y[n-2]`
    #[inline]
    pub fn process_single(&self, x_0: f64, x_1: f64, y_1: f64, y_2: f64) -> f64 {
        self.x_0_const * x_0 + self.x_1_const *x_1 + self.re_2 * y_1 - self.mag_sq * y_2
    }

    /// Returns the locations of the two poles in the filter
    /// The first returned pole will have a positive imaginary component.
    #[inline]
    pub fn get_pole_locs(&self) -> (Complex<f64>, Complex<f64>) {
        let theta = (self.re_2 / 2.0 / self.mag_sq.sqrt()).acos();
        (Complex::from_polar(self.mag_sq.sqrt(), theta), Complex::from_polar(self.mag_sq.sqrt(), -theta))
    }
}

impl BodePlotTransferFunction for PhasedConjPoleResonator {
    fn get_value(&self, z: Complex<f64>) -> Complex<f64> {
        let poles = self.get_pole_locs();
        ((z - poles.0) * (z - poles.1)).inv().scale(self.gain)
    }
}

impl LTISystem for PhasedConjPoleResonator {
    fn process(&self, samples: &Vec<f64>) -> Vec<f64> {
        let mut buf = vec![0.0; samples.len()];
        self.process_buf(&samples[..], &mut buf[..]);
        buf
    }
}

#[cfg(test)]
mod tests {
    use std::{time::Instant, f64::consts::PI};

    use bode_plot::{create_generic_plot, create_plot, DEFAULT_HEIGHT, DEFAULT_WIDTH};
    use wav_util::*;

    use super::*;

    #[test]
    fn test_resonator_sections() {
        let resonator = PhasedConjPoleResonator::new_polar(0.9999, PI / 32.0, 0.0, 1.0);
        let ([chan1, chan2], _) = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = Instant::now();
        
        resonator.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        resonator.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_phased_harmonic_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&resonator as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_impulse_response() {
        let resonator = PhasedConjPoleResonator::new_polar(0.9999, PI / 512.0, PI, 1.0);
        let mut impulse = vec![0.0; 10000];
        impulse[0] = 1.0;
        let mut response = vec![0.0; impulse.len()];
        resonator.process_buf(&impulse[..], &mut response[..]);

        create_generic_plot(
            "Impulse Response".into(),
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            response.iter().enumerate().map(|(i, v)| (i as f64, *v)).collect(),
            0.0..10000.0,
            -1.0..1.0)
            .unwrap();

        let right_channel: Vec<f64> = response.iter().map(|v| *v).collect();
        let left_channel = right_channel.clone();
        write_wave([left_channel, right_channel], "audio/test_impulse_response.wav", 48_000).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}