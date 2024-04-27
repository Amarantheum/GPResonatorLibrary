//! This library contains functions for building a resonator array from audio data.
//! [`scaled_builder::ScaledResonatorPlanner`] can be found in the scaled_builder module.
use bode_plot::{create_plot, DEFAULT_HEIGHT, DEFAULT_WIDTH, plot::BodePlotTransferFunction};
use fft::FftCalculator;
use gp_resonator::{resonator_array::ConjPoleResonatorArray, resonator::ConjPoleResonator};
use nalgebra::{DMatrix, LU, DVector};
use std::{error::Error, f64::consts::PI};

use crate::{fft::window::{Rectangular, WindowFunction}, resonance_estimator::ResEstimator};
use statrs::statistics::Statistics;
use find_peaks::{PeakFinder, Peak};

mod resonance_estimator;
mod simple_builder;
mod gradient_descent_builder;
mod expectation_maximization;
mod resonance_params;
mod approx_gradient_descent_builder;
pub mod fft;
pub mod scaled_builder;

fn find_peak_left_bound(peak: &Peak<f64>, neighbor: Option<&Peak<f64>>, data: &[f64]) -> usize {
    let mut cur_index = peak.middle_position() - 10;
    let mut prev_index = peak.middle_position();
    while cur_index > 0 && data[cur_index] < data[prev_index] {
        cur_index -= 1;
        prev_index -= 1;
    }
    cur_index
}

fn find_peak_right_bound(peak: &Peak<f64>, neighbor: Option<&Peak<f64>>, data: &[f64]) -> usize {
    let mut cur_index = peak.middle_position() + 10;
    let mut prev_index = peak.middle_position();
    while cur_index < data.len() - 1 && data[cur_index] < data[prev_index]  {
        cur_index += 1;
        prev_index += 1;
    }
    cur_index
}

// TODO: bug when setting min_freq to something other than 0.0 (frequency shifted)
// max_num_peaks unused
fn audio_to_resonator_array(audio: &[f64], sample_rate: f64, max_num_peaks: usize, min_freq: f64, max_freq: f64) -> Result<ConjPoleResonatorArray, Box<dyn Error>> {
    assert!(audio.len() > 0);
    let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
    let spec_size = 2_usize.pow(near_pow_2 as u32);
    let mut calc = FftCalculator::new(audio.len(), spec_size - audio.len()).unwrap(); // shouldn't fail
    let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
    let log_spectrum = spectrum.iter().map(|v| v.log10()).collect::<Vec<f64>>();
    let min_bin = (min_freq / sample_rate * (calc.zero_pad_length + calc.size) as f64).floor() as usize;
    let max_bin = (max_freq / sample_rate * (calc.zero_pad_length + calc.size) as f64).floor() as usize;
    let spec_slice = &spectrum[min_bin..max_bin];
    let log_spec_slice = &log_spectrum[min_bin..max_bin];

    bode_plot::create_generic_plot("spectrum".into(), 4000, 800, log_spec_slice.iter().enumerate().map(|v| (v.0 as f64, *v.1)).collect::<Vec<(f64, f64)>>(), min_bin as f64..max_bin as f64, -5.0..log_spec_slice.max())?;
    // TODO: custom peak finding algorithm for performance
    let mut peaks = PeakFinder::new(log_spec_slice)
        .with_min_prominence(1.0)
        .with_min_distance(10)
        .find_peaks();
    // sort peaks by position on x axis
    peaks.sort_by(|a, b| a.middle_position().partial_cmp(&b.middle_position()).unwrap());

    let mut peak_points: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(peaks.len());
    let mut iter_prev = peaks.iter();
    let iter_cur = peaks.iter();
    let mut iter_next = peaks.iter();
    iter_next.next();

    let mut prev = None;
    let mut next = iter_next.next();
    for peak in iter_cur {
        let left = find_peak_left_bound(peak, prev, log_spec_slice);
        let right = find_peak_right_bound(peak, next, log_spec_slice);

        let mut x = Vec::with_capacity(right - left);
        let mut y = Vec::with_capacity(right - left);
        for i in left..=right {
            x.push(i as f64 / (spec_size >> 1) as f64 * PI);
            y.push(spec_slice[i]);
        }
        peak_points.push((x, y));

        prev = iter_prev.next();
        next = iter_next.next();
    }
    println!("GOT: {} peaks", peaks.len());
    // for r in &peak_points {
    //     let scatter = r.0.iter().zip(r.1.iter()).map(|v| (*v.0, *v.1)).collect::<Vec<(f64, f64)>>();
    //     bode_plot::create_generic_plot_scatter("spectrum".into(), 800, 800, vec![], scatter, r.0[0]..r.0[r.0.len() - 1], 0.0..(&r.1).max())?;
    // }

    let mut resonator_params = Vec::with_capacity(peaks.len());
    // vector we use to solve system later (Ax = b)
    let mut b = Vec::with_capacity(peaks.len());
    for ((x, y), p) in peak_points.into_iter().zip(peaks.iter()) {
        let resonance_estimator = ResEstimator::new(x.clone(), y.clone(), p.middle_position() as f64 / (spec_size >> 1) as f64 * PI);
        b.push(spectrum[p.middle_position()]);
        match resonance_estimator.solve() {
            Ok(v) => resonator_params.push(v.into_res_params()),
            Err(e) => {
                match e {
                    resonance_estimator::SolveError::LostPatience(v) => {
                        // ignore lost patience errors for now since they still have low objective functions
                        resonator_params.push(v.into_res_params());
                    },
                    _ => return Err(e.into()),
                }
            }
        }
    }

    debug_assert!(
        {
            let mut out = true;
            for params in &resonator_params {
                if params.g < 0.0 || params.r < 0.0 || params.r >= 1.0 {
                    out = false;
                }
            }
            out
        }
    );

    let mut system_arr = Vec::with_capacity(resonator_params.len().pow(2));
    for p1 in &resonator_params {
        for p2 in &resonator_params {
            // effect of p1 on p2
            system_arr.push(p2.predict_r_influence(p1.w_0))
        }
    }

    let scatter = peaks.into_iter().map(|v| (v.middle_position() as f64, v.height.unwrap())).collect::<Vec<(f64, f64)>>();

    let spectrum_plot = spectrum[min_bin..max_bin].iter().zip(min_bin..max_bin).map(|(f, u)| (u as f64, *f)).collect::<Vec<(f64, f64)>>();
    bode_plot::create_generic_plot_scatter("spectrum".into(), 4000, 800, spectrum_plot, scatter, min_bin as f64..max_bin as f64, -5.0..spectrum.max())?;
    
    // set up system
    let system = DMatrix::from_row_slice(resonator_params.len(), resonator_params.len(), &system_arr);
    let lu = LU::new(system);
    // solve system
    let soln: Vec<f64> = lu.solve(&DVector::from_vec(b)).ok_or("failed to solve system")?.data.into();
    
    // build resonator array
    let mut resonator_array = ConjPoleResonatorArray::new(48_000_f64, soln.len());
    for (g, params) in soln.iter().zip(resonator_params.iter()) {
        resonator_array.add_resonator_raw(ConjPoleResonator::new_polar(params.r, params.w_0, *g * params.w_0.sin()));
    }

    Ok(resonator_array)
    // println!("SIZE: {}", resonator_params.len());
    // let scatter = peaks.into_iter().map(|v| (v.middle_position() as f64, v.height.unwrap())).collect::<Vec<(f64, f64)>>();

    // let spectrum_plot = spectrum[min_bin..max_bin].iter().zip(min_bin..max_bin).map(|(f, u)| (u as f64, *f)).collect::<Vec<(f64, f64)>>();
    // bode_plot::create_generic_plot_scatter("spectrum".into(), 4000, 800, spectrum_plot, scatter, min_bin as f64..max_bin as f64, -5.0..spectrum.max())?;
    // create_plot("TEST PLOT".into(), 4000, 800, vec![&resonator_array as &dyn BodePlotTransferFunction]).unwrap();
    // loop {}
    // //std::thread::sleep(std::time::Duration::from_millis(10000));
    // Err("not implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wav_util::*;

    #[test]
    fn test_find_peaks() {
        let ([chan1, chan2], sample_rate) = read_wave("./tests/piano.wav").unwrap();
        let mut array = audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 0.0, sample_rate as f64 / 2.0).unwrap();
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
        let mut array = audio_to_resonator_array(&chan1[..], sample_rate as f64, 100, 0.0, sample_rate as f64 / 2.0).unwrap();
        
        let mut impulse = vec![0_f64; 48_000 * 10];
        impulse[0] = 1.0;
        let mut out = vec![0_f64; 48_000 * 10];

        let start = std::time::Instant::now();
        array.process_buf(&impulse, &mut out[..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());

        write_wave([out.clone(), out], "./tests/test_fm_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
