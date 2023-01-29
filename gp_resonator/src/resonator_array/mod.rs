use std::f64::consts::PI;

use bode_plot::plot::BodePlotTransferFunction;
use num_complex::Complex;

use crate::resonator::{self, ConjPoleResonator};

#[derive(Debug, Clone)]
struct ConjPoleResonatorState {
    x_1: f64,
    y_1: f64,
    y_2: f64,
}

impl Default for ConjPoleResonatorState {
    fn default() -> Self {
        Self {
            x_1: 0_f64,
            y_1: 0_f64,
            y_2: 0_f64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConjPoleResonatorArray {
    sample_rate: f64,
    resonators: Vec<(ConjPoleResonatorState, ConjPoleResonator)>
}

impl ConjPoleResonatorArray {
    #[inline]
    pub fn new(sample_rate: f64, capacity: usize) -> Self {
        Self {
            sample_rate,
            resonators: Vec::with_capacity(capacity),
        }
    }

    /// Process a single data point given the required values from the difference equation
    /// and returns `y[n]`. Prefer [`Self::process_buf`] for better cache performance.
    /// # Arguments
    /// * `x_1` - value of `x[n-1]`
    /// * `y_1` - value of `y[n-1]`
    /// * `y_2` - value of `y[n-2]`
    #[inline]
    pub fn process_single(&self, x_1: f64, y_1: f64, y_2: f64) -> f64 {
        let mut total = 0_f64;
        for r in &self.resonators {
            total += r.1.process_single(x_1, y_1, y_2);
        }
        total
    }

    #[inline]
    pub fn process_buf(&mut self, x: &[f64], buf: &mut [f64]) {
        debug_assert!(buf.len() > 0 && x.len() == buf.len());
        let mut workspace = Vec::<f64>::with_capacity(x.len() + 4);
        unsafe { workspace.set_len(x.len() + 4); }
        for (state, res) in &mut self.resonators {
            workspace[0] = state.y_2;
            workspace[1] = state.y_1;
            workspace[2] = res.process_single(state.x_1, state.y_1, state.y_2);
            buf[0] += workspace[2];
            for i in 3..buf.len() + 1 {
                workspace[i] = res.process_single(x[i - 1 - 2], workspace[i - 1], workspace[i - 2]);
                buf[i - 2] += workspace[i];
            }
            state.x_1 = x[buf.len() - 1];
            state.y_1 = workspace[buf.len() + 1];
            state.y_2 = workspace[buf.len()];
        }
    }

    #[inline]
    pub fn reset_state(&mut self) {
        for (state, _) in &mut self.resonators {
            *state = ConjPoleResonatorState::default();
        }
    }

    pub fn add_resonator(&mut self, freq: f64, intensity: f64, gain: f64) -> Result<(), &'static str> {
        if freq > self.sample_rate / 2.0 {
            return Err("frequency exceeds the nyquist limit")
        }
        let arg = 2.0 * PI * freq / self.sample_rate;
        self.resonators.push((ConjPoleResonatorState::default(), ConjPoleResonator::new_polar(intensity, arg, gain)));
        Ok(())
    }
}

impl BodePlotTransferFunction for ConjPoleResonatorArray {
    fn get_value(&self, z: num_complex::Complex<f64>) -> num_complex::Complex<f64> {
        let mut total = Complex::new(0.0, 0.0);
        for (_, res) in &self.resonators {
            total += res.get_value(z);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use bode_plot::{create_plot, DEFAULT_WIDTH, DEFAULT_HEIGHT};

    use super::*;
    use crate::wav_util::*;

    #[test]
    fn test_add_resonator() {
        let mut array = ConjPoleResonatorArray::new(48_000.0, 1);
        array.add_resonator(440.0, 0.9999, 1.0);

        let [chan1, chan2] = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        array.process_buf(&chan1[..], &mut out_chan1[..]);
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_add_resonator.wav", 48_000).unwrap();
    }

    #[test]
    fn test_harmonic_resonator() {
        let mut array = ConjPoleResonatorArray::new(48_000.0, 1);
        for i in 1..54 * 8 + 1 {
            array.add_resonator(55.0 * i as f64, 0.9999, 1.0).unwrap();
        }
        let [chan1, chan2] = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_harmonic_resonator.wav", 48_000).unwrap();
        create_plot("Harmoincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}