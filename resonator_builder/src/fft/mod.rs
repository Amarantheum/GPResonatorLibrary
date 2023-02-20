use std::sync::Arc;
use rustfft::algorithm::Radix4;
use std::error::Error;
use rustfft::num_complex::Complex;
use rustfft::Fft;
use window::RealWindowFn;

pub mod window;

/// A structure for calculating FFTs
pub struct FftCalculator {
    fft_planner: Radix4<f64>,
    // the size passed to create the fft planner
    pub size: usize,
    pub zero_pad_length: usize,
    internal_buf: Vec<f64>
}

impl FftCalculator {
    /// Construct a new Fft calculator with given size and zero pad length
    /// # Arguments
    /// * `size` - the size of the fft to calculate
    /// * `zero_pad_length` - the amount of zero padding to add to end of an input signal
    /// # Note: `size` + `zero_pad_length` should be a power of 2.
    pub fn new(size: usize, zero_pad_length: usize) -> Result<Self, Box<dyn Error>> {
        if !(size + zero_pad_length).is_power_of_two() {
            return Err(format!("failed to create fft calculator because size {} and zero_pad_length {} do not add to power of 2", size, zero_pad_length).into());
        }
        Ok(
            Self {
                fft_planner: Radix4::new(size + zero_pad_length, rustfft::FftDirection::Forward),
                size,
                zero_pad_length,
                internal_buf: vec![0_f64; size],
            }
        )
    }

    /// Computes the fft of the given real-valued signal
    pub fn real_fft(&mut self, samples: &[f64], window_fn: RealWindowFn) -> Vec<Complex<f64>>
    {
        assert!(samples.len() == self.size);
        for i in 0..samples.len() {
            self.internal_buf[i] = samples[i];
        }
        window_fn(&mut self.internal_buf[..]);
        let mut out = Vec::with_capacity(self.size + self.zero_pad_length);
        for i in 0..samples.len() {
            out.push(Complex::<f64>::new(self.internal_buf[i], 0.0));
        }
        for _ in 0..self.zero_pad_length {
            out.push(Complex::<f64>::new(0.0, 0.0));
        }
        self.fft_planner.process(&mut out[..]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{*, window::{Rectangular, WindowFunction}};
    use statrs::statistics::Statistics;
    use rand::prelude::*;

    #[test]
    fn test_zero_padding() {
        let mut calc1 = FftCalculator::new(1024, 0).unwrap();
        let mut calc2 = FftCalculator::new(1024, 1024).unwrap();
        let mut calc3 = FftCalculator::new(1024, 1024 * 3).unwrap();
        let mut rng = thread_rng();
        let signal = (0..1024).map(|_| rng.gen_range(-1.0..1.0)).collect::<Vec<f64>>();
        let res1 = calc1.real_fft(&signal[..], Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
        let res2 = calc2.real_fft(&signal[..], Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
        let res3 = calc3.real_fft(&signal[..], Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
        println!("res1: {}", res1.mean());
        println!("res2: {}", res2.mean());
        println!("res3: {}", res3.mean());
    }
}