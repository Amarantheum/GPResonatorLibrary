use std::f64::consts::PI;
use super::ConjPoleResonatorState;
use bode_plot::plot::BodePlotTransferFunction;
use num_complex::Complex;

use crate::resonators::PhasedConjPoleResonator;

/// A resonator composed of a dynamic number of [`ConjPoleResonator`]s.
/// See [`PhasedResonatorArray::process_buf`].
#[derive(Debug, Clone)]
pub struct PhasedResonatorArray {
    /// The sample rate of audio being passed through the resonator
    pub sample_rate: f64,
    resonators: Vec<(ConjPoleResonatorState, PhasedConjPoleResonator)>
}

impl PhasedResonatorArray {
    /// Creates a new instance of [`PhasedResonatorArray`] with given sample rate
    /// and capacity. The capacity passed is only used to allocating the initial memory
    /// for the resonator array and is mainly for efficiency (any value can be passed).
    /// # Arguments
    /// * `sample_rate` - the sample rate of the audio being passed to the system (used to determine pole locations given a frequency in Hz)
    /// * `capacity` - a soft capacity size for the resonator array
    #[inline]
    pub fn new(sample_rate: f64, capacity: usize) -> Self {
        Self {
            sample_rate,
            resonators: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of resonators in the [`PhasedResonatorArray`]
    #[inline]
    pub fn size(&self) -> usize {
        self.resonators.len()
    }

    /// Processes a single data point given the required values from the difference equation
    /// and returns `y[n]`. Prefer [`Self::process_buf`] for better cache performance.
    /// # Arguments
    /// * `x_1` - value of `x[n-1]`
    /// * `y_1` - value of `y[n-1]`
    /// * `y_2` - value of `y[n-2]`
    #[inline]
    pub fn process_single(&self, x_0: f64, x_1: f64, y_1: f64, y_2: f64) -> f64 {
        let mut total = 0_f64;
        for r in &self.resonators {
            total += r.1.process_single(x_0, x_1, y_1, y_2);
        }
        total
    }

    /// Takes a signal and adds the output of the signal going through the resonator array to a given buffer.
    /// 
    /// Calling this updates the resonator array so that the next call will be treated as if the given signal is concatenated
    /// to the previous signal. To prevent this, one may call [`Self::reset_state`].
    /// # Arguments
    /// * `x` - signal buffer
    /// * `buf` - buffer to write to
    #[inline]
    pub fn process_buf(&mut self, x: &[f64], buf: &mut [f64]) {
        if x.len() == 0 {
            return;
        }
        let mut workspace = Vec::<f64>::with_capacity(x.len() + 4);
        unsafe { workspace.set_len(x.len() + 4); }
        for (state, res) in &mut self.resonators {
            workspace[0] = state.y_2;
            workspace[1] = state.y_1;
            workspace[2] = res.process_single(x[0], state.x_1, state.y_1, state.y_2);
            buf[0] += workspace[2];
            for i in 3..buf.len() + 2 {
                workspace[i] = res.process_single(x[i - 2], x[i - 1 - 2], workspace[i - 1], workspace[i - 2]);
                buf[i - 2] += workspace[i];
            }
            state.x_1 = x[buf.len() - 1];
            state.y_1 = workspace[buf.len() + 1];
            state.y_2 = workspace[buf.len()];
        }
    }

    /// Resets the state of the internal resonators. The state of the internal
    /// resonators is used to allow continuous signal processing in given chunks.
    /// See [`Self::process_buf`].
    #[inline]
    pub fn reset_state(&mut self) {
        for (state, _) in &mut self.resonators {
            *state = ConjPoleResonatorState::default();
        }
    }

    /// Add a resonator with given intensity and gain to the array.
    /// # Arguments
    /// * `freq` - the frequency of the resonator in Hz
    /// * `decay` - the time in seconds for the signal to reach 10% amplitude
    /// * `gain` - the gain of the resonator output (amplitude)
    #[inline]
    pub fn add_resonator(&mut self, freq: f64, decay: f64, phase: f64, gain: f64) -> Result<(), &'static str> {
        let arg = 2.0 * PI * freq / self.sample_rate;
        self.add_resonator_theta(arg, decay, phase, gain)
    }

    /// Add a resonator with given intensity and gain to the array.
    /// # Arguments
    /// * `theta` - the frequency of the resonator in radians
    /// * `decay` - the time in seconds for the signal to reach 10% amplitude
    /// * `gain` - the gain of the resonator output (amplitude)
    #[inline]
    pub fn add_resonator_theta(&mut self, arg: f64, decay: f64, phase: f64, gain: f64) -> Result<(), &'static str> {
        if arg > PI {
            return Err("frequency exceeds the nyquist limit")
        }
        if arg < 0.0 {
            return Err("frequency must be positive")
        }
        let r = Self::decay_to_mag(self.sample_rate, decay);
        // normalize gain
        let gain = gain * r * arg.sin();
        self.resonators.push((ConjPoleResonatorState::default(), PhasedConjPoleResonator::new_polar(r, arg, phase, gain)));
        Ok(())
    }

    // calculate the correct magnitude for a pole to get the desired decay
    #[inline]
    fn decay_to_mag(sr: f64, decay: f64) -> f64 {
        let decay_samples = decay * sr;
        0.1_f64.powf(1.0 / decay_samples)
    }

    /// Updates each resonator in the array by passing each resonator into the given function
    /// # Arguments
    /// * `f` - A function the operates on each [`ConjPoleResonator`] in the [`PhasedResonatorArray`]
    #[inline]
    pub fn update_resonators<F: FnMut(usize, &mut PhasedConjPoleResonator)>(&mut self, mut f: F) {
        for (i, (_, resonator)) in self.resonators.iter_mut().enumerate() {
            f(i, resonator)
        }
    }

    /// Sets the decay amount for each resonator in the array to the given value.
    /// # Arguments
    /// * `decay` - The decay amount to set each resonator to
    #[inline]
    pub fn set_resonator_decays(&mut self, decay: f64) {
        let sr = self.sample_rate;
        for (_, resonator) in self.resonators.iter_mut() {
            resonator.set_mag(Self::decay_to_mag(sr, decay))
        }
    }
    
    /// Add a pre-built resonator to the current resonator array.
    /// User must ensure that the resonator frequency is what they expect since adding resonators using this method doesn't factor in sample rate.
    /// # Arrguments
    /// * `resonator` - The resonator to be added to the resonator array
    pub fn add_resonator_raw(&mut self, resonator: PhasedConjPoleResonator) {
        self.resonators.push((ConjPoleResonatorState::default(), resonator))
    }
}

impl BodePlotTransferFunction for PhasedResonatorArray {
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
    use wav_util::*;

    use super::*;

    #[test]
    fn test_add_resonator() {
        let mut array = PhasedResonatorArray::new(48_000.0, 1);
        array.add_resonator(440.0, 1.0, PI / 2.0, 1.0).unwrap();

        let ([chan1, chan2], _) = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        array.process_buf(&chan1[..], &mut out_chan1[..]);
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_add_resonator.wav", 48_000).unwrap();
    }

    #[test]
    fn test_harmonic_resonator() {
        let mut array = PhasedResonatorArray::new(48_000.0, 1);
        for i in 1..54 * 8 + 1 {
            array.add_resonator(55.0 * i as f64, PI / 2.0,0.1, 1.0).unwrap();
        }
        let ([chan1, chan2], _) = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_harmonic_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_harmonic_resonator_sections() {
        let mut array = PhasedResonatorArray::new(48_000.0, 1);
        for i in 1..54 * 8 + 1 {
            array.add_resonator(55.0 * i as f64, 0.1, PI / 2.0,1.0).unwrap();
        }
        let ([chan1, chan2], _) = read_wave("audio/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = Instant::now();
        let middle = chan1.len() / 2;
        array.process_buf(&chan1[..middle], &mut out_chan1[..middle]);
        array.process_buf(&chan1[middle..], &mut out_chan1[middle..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "audio/test_harmonic_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_harmonic_resonator_basic() {
        let mut array = PhasedResonatorArray::new(48_000.0, 1);
        for i in 1..2 {
            array.add_resonator(55.0 * i as f64, 0.1, PI / 2.0,1.0).unwrap();
        }

        let input = vec![0.0, 1.0, 0.0, -0.5, -0.7, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, -0.5, -0.7, 1.0, 0.0, 0.5];
        let mut out1 = vec![0.0; input.len()];
        array.process_buf(&input[..], &mut out1[..]);

        array.reset_state();
        let mut out2 = vec![0.0; input.len()];
        array.process_buf(&input[..4], &mut out2[..4]);
        array.process_buf(&input[4..8], &mut out2[4..8]);
        array.process_buf(&input[8..12], &mut out2[8..12]);
        array.process_buf(&input[12..], &mut out2[12..]);
        
        assert!(out1 == out2);
    }

    #[test]
    fn test_decay_math() {
        // decay = 10 samples
        let decay = 1_f64 * 48_000_f64;
        let r = 0.00001_f64.powf(1.0 / decay);
        println!("{}", r);
    }
}