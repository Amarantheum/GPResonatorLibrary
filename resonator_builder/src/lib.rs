use fft::FftCalculator;
use gp_resonator::resonator_array::ConjPoleResonatorArray;
use std::error::Error;

use crate::fft::window::{Rectangular, WindowFunction};
use statrs::statistics::Statistics;

mod resonance_estimator;
mod fft;

pub fn audio_to_resonator_array(audio: &[f64], sample_rate: f64, max_num_peaks: f64, min_freq: f64, max_freq: f64) -> Result<ConjPoleResonatorArray, Box<dyn Error>> {
    assert!(audio.len() > 0);
    let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
    let mut calc = FftCalculator::new(audio.len(), near_pow_2 - audio.len()).unwrap(); // shouldn't fail
    let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();

    let spectrum_plot = spectrum.iter().zip(0..spectrum.len()).map(|(f, u)| (*f, u as f64)).collect::<Vec<(f64, f64)>>();
    bode_plot::create_generic_plot("spectrum".into(), 1920, 1080, spectrum_plot, 0.0..(spectrum.len() as f64), 0.0..spectrum.max())?;
    std::thread::sleep(std::time::Duration::from_micros(10000));
    Err("not implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_peaks() {
        //audio_to_resonator_array(audio, sample_rate, max_num_peaks, min_freq, max_freq)
    }
}
