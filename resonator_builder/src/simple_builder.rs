use crate::fft::{FftCalculator, window::{Rectangular, WindowFunction}};
use gp_resonator::{resonator_array::ConjPoleResonatorArray, resonator};
use std::{error::Error, f64::consts::PI};
use find_peaks::{PeakFinder, Peak};


/// Converts the index in the log_spec_slice spectrum to a frequency
/// 
fn slice_index_to_freq(log_spec_index: usize, min_bin: usize, spec_size: usize, sample_rate: f64) -> f64 {
    (log_spec_index + min_bin) as f64 / spec_size as f64 * sample_rate
}

pub fn basic_audio_to_resonator_array(audio: &[f64], sample_rate: f64, max_num_peaks: usize, min_freq: f64, max_freq: f64) -> Result<ConjPoleResonatorArray, Box<dyn Error>> {
    assert!(audio.len() > 0);
    let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
    let spec_size = 2_usize.pow(near_pow_2 as u32);
    let mut calc = FftCalculator::new(audio.len(), spec_size - audio.len()).unwrap(); // shouldn't fail
    let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
    let log_spectrum = spectrum.iter().map(|v| v.log10()).collect::<Vec<f64>>();
    let min_bin = (min_freq / sample_rate * spec_size as f64).floor() as usize;
    let max_bin = (max_freq / sample_rate * spec_size as f64).floor() as usize;
    let log_spec_slice = &log_spectrum[min_bin..max_bin];

    //bode_plot::create_generic_plot("spectrum".into(), 4000, 800, log_spec_slice.iter().enumerate().map(|v| (v.0 as f64, *v.1)).collect::<Vec<(f64, f64)>>(), min_bin as f64..max_bin as f64, -5.0..log_spec_slice.max())?;
    let peaks = PeakFinder::new(log_spec_slice)
        .with_min_prominence(2.0)
        .with_min_distance(10)
        .find_peaks();

    let mut resonator_array = ConjPoleResonatorArray::new(48_000_f64, peaks.len());
    for peak in peaks {
        //println!("PEAK: {}", slice_index_to_freq(peak.middle_position(), min_bin, spec_size) / PI * 180.0);
        resonator_array.add_resonator(slice_index_to_freq(peak.middle_position(), min_bin, spec_size, sample_rate), 1.0, 1.0)?;
    }

    Ok(resonator_array)
}

pub fn scaled_audio_to_resonator_array(audio: &[f64], sample_rate: f64, max_num_peaks: usize, min_freq: f64, max_freq: f64) -> Result<ConjPoleResonatorArray, Box<dyn Error>> {
    assert!(audio.len() > 0);
    let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
    let spec_size = 2_usize.pow(near_pow_2 as u32);
    let mut calc = FftCalculator::new(audio.len(), spec_size - audio.len()).unwrap(); // shouldn't fail
    let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
    let log_spectrum = spectrum.iter().map(|v| v.log10()).collect::<Vec<f64>>();
    let min_bin = (min_freq / sample_rate * spec_size as f64).floor() as usize;
    let max_bin = (max_freq / sample_rate * spec_size as f64).floor() as usize;
    let spec_slice = &spectrum[min_bin..max_bin];
    
    let mut spec_max = 0_f64;
    for v in spec_slice {
        if *v > spec_max {
            spec_max = *v;
        }
    }
    let log_spec_slice = &log_spectrum[min_bin..max_bin];

    //bode_plot::create_generic_plot("spectrum".into(), 4000, 800, log_spec_slice.iter().enumerate().map(|v| (v.0 as f64, *v.1)).collect::<Vec<(f64, f64)>>(), min_bin as f64..max_bin as f64, -5.0..log_spec_slice.max())?;
    let peaks = PeakFinder::new(log_spec_slice)
        .with_min_prominence(2.0)
        .with_min_distance(10)
        .find_peaks();

    let mut resonator_array = ConjPoleResonatorArray::new(48_000_f64, peaks.len());
    for peak in peaks {
        let bin = peak.middle_position();
        resonator_array.add_resonator(slice_index_to_freq(bin, min_bin, spec_size, sample_rate), 1.0, spec_slice[bin] / spec_max)?;
    }

    Ok(resonator_array)
}






#[cfg(test)]
mod tests {
    use super::*;
    use wav_util::*;
    use bode_plot::{create_plot, DEFAULT_HEIGHT, DEFAULT_WIDTH, plot::BodePlotTransferFunction};

    #[test]
    fn test_find_peaks() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/piano.wav").unwrap();
        let mut array = scaled_audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 0.0, sample_rate as f64 / 2.0).unwrap();
        let ([chan1, chan2], _) = read_wave("./tests/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = std::time::Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "./tests/test_piano_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_find_peaks_fm() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/fm.wav").unwrap();
        let mut array = scaled_audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 500.0, sample_rate as f64 / 2.0).unwrap();
        let ([chan1, chan2], _) = read_wave("./tests/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = std::time::Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "./tests/test_fm_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_reverb() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/reverb_ir.wav").unwrap();
        let mut array = scaled_audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 500.0, sample_rate as f64 / 2.0).unwrap();
        let ([chan1, chan2], _) = read_wave("./tests/cool_tune.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = std::time::Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "./tests/test_reverb_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_atmosphere() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/Datmosphere.wav").unwrap();
        let mut array = scaled_audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 500.0, sample_rate as f64 / 2.0).unwrap();
        let ([chan1, chan2], _) = read_wave("./tests/beep2.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = std::time::Instant::now();
        array.process_buf(&chan1[..], &mut out_chan1[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "./tests/test_d_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}