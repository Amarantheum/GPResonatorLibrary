use fft::FftCalculator;
use gp_resonator::resonator_array::ConjPoleResonatorArray;
use std::error::Error;

use crate::fft::window::{Rectangular, WindowFunction};
use statrs::statistics::Statistics;
use find_peaks::{PeakFinder, Peak};

mod resonance_estimator;
mod fft;

pub fn audio_to_resonator_array(audio: &[f64], sample_rate: f64, max_num_peaks: usize, min_freq: f64, max_freq: f64) -> Result<ConjPoleResonatorArray, Box<dyn Error>> {
    assert!(audio.len() > 0);
    let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
    let mut calc = FftCalculator::new(audio.len(), 2_usize.pow(near_pow_2 as u32) - audio.len()).unwrap(); // shouldn't fail
    let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
    let min_bin = (min_freq / sample_rate * (calc.zero_pad_length + calc.size) as f64).floor() as usize;
    let max_bin = (max_freq / sample_rate * (calc.zero_pad_length + calc.size) as f64).floor() as usize;
    let spec_slice = &spectrum[min_bin..max_bin];

    let peaks = PeakFinder::new(spec_slice)
        .with_min_prominence(10.0)
        .with_min_height(0.0)
        .with_min_distance(10)
        .find_peaks();
    let scatter = peaks.into_iter().map(|v| (v.middle_position() as f64, v.height.unwrap())).collect::<Vec<(f64, f64)>>();
    

    let spectrum_plot = spectrum[min_bin..max_bin].iter().zip(min_bin..max_bin).map(|(f, u)| (u as f64, *f)).collect::<Vec<(f64, f64)>>();
    bode_plot::create_generic_plot_scatter("spectrum".into(), 1920, 800, spectrum_plot, scatter, min_bin as f64..max_bin as f64, 0.0..spectrum.max())?;
    loop {}
    //std::thread::sleep(std::time::Duration::from_millis(10000));
    Err("not implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wav_util::*;
    use statrs::statistics::Statistics;

    #[test]
    fn test_find_peaks() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/piano.wav").unwrap();
        audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 0.0, sample_rate as f64 / 16.0).unwrap();
    }
}
