//! This crate contains abstractions for creating general purpose resonators.
//! A general purpose resonator is an array of resonators that can be used to create resonances at arbitrary frequencies for real-time audio.
//! You are probably looking for [`crate::resonator_array::ConjPoleResonatorArray`].

pub mod resonator;
pub mod resonator_array;
mod wav_util;

#[cfg(test)]
mod tests {
    use bode_plot::{create_log_plot, create_plot, plot::{BodePlotTransferFunction, LTISystem}, DEFAULT_WIDTH, DEFAULT_HEIGHT, create_simulation_plot}; 
    use num_complex::Complex;
    use std::{error::Error, f64::consts::PI};
    use crate::resonator::ConjPoleResonator;
    use statrs::statistics::*;

    #[test]
    fn test_single_bode_plot() {
        let test_transfer1 = ConjPoleResonator::new_polar(0.9, 3.14159 / 4.0, 1.0);
        create_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_transfer1 as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_basic_bode_plot() {
        let test_transfer1 = ConjPoleResonator::new_polar(0.5, 3.14159 / 2.0, 1.0);
        let test_transfer2 = ConjPoleResonator::new_polar(0.5, 3.14159 / 4.0, 1.0);
        let test_transfer3 = ConjPoleResonator::new_polar(0.5, 3.14159 / 4.0 * 3.0, 1.0);
        create_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_transfer1 as &dyn BodePlotTransferFunction, &test_transfer2 as &dyn BodePlotTransferFunction, &test_transfer3 as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1000));
    }

    #[test]
    fn test_simulation_plot() {
        let test_system1 = ConjPoleResonator::new_polar(0.9, 3.14159 / 2.0, 1.0);
        let test_system2 = ConjPoleResonator::new_polar(0.5, 3.14159 / 4.0, 1.0);
        let test_system3 = ConjPoleResonator::new_polar(0.5, 3.14159 / 4.0 * 3.0, 1.0);
        create_simulation_plot("TEST SIM".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_system1, &test_system2, &test_system3].into_iter().map(|v| v as &dyn LTISystem).collect()).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_wav() {
        let mut reader = hound::WavReader::open("./audio/test_noise.wav").unwrap();
        let samples = reader.samples::<f32>()
            .map(|v| {
                v.unwrap()
            })
            .collect::<Vec<f32>>();
        let mut chan1 = Vec::with_capacity(samples.len() / 2);
        let mut chan2 = Vec::with_capacity(samples.len() / 2);
        for i in 0..samples.len() {
            if i % 2 == 0 {
                chan1.push(samples[i] as f64)
            } else {
                chan2.push(samples[i] as f64)
            }
        }

        let test_system1 = ConjPoleResonator::new_polar(0.999999, PI / 128.0, 1.0);
        let mut chan1 = test_system1.process(&chan1);
        let mut chan2 = test_system1.process(&chan2);

        let mut max = f64::MIN;
        for f in chan1.iter().chain(chan2.iter()) {
            if f.abs() > max {
                max = f.abs();
            }
        }
        for f in chan1.iter_mut().chain(chan2.iter_mut()) {
            *f /= max;
        }
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create("./audio/filtered_noise.wav", spec).unwrap();
        for s in chan1.into_iter().zip(chan2.into_iter()) {
            writer.write_sample(s.0 as f32).unwrap();
            writer.write_sample(s.1 as f32).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_ir() {
        let test_system1 = ConjPoleResonator::new_polar(0.9999, PI / 128.0, 0.001);
        let mut impulse = vec![0_f64; 100_000];
        impulse[0] = 1.0;
        let ir = test_system1.process(&impulse);
        let max = (&ir).max();
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create("./audio/ir.wav", spec).unwrap();
        for s in ir {
            writer.write_sample((s / max) as f32).unwrap();
            writer.write_sample((s / max) as f32).unwrap();
        }
        writer.finalize().unwrap();
    }
}
