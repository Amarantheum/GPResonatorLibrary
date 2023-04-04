use std::{f64::consts::PI};
use crate::fft::{FftCalculator, window::{Rectangular, WindowFunction}};
use find_peaks::{PeakFinder};
use gp_resonator::resonator_array::ConjPoleResonatorArray;

#[derive(Debug, Clone)]
pub struct ScaledResonatorPlan {
    /// Each value corresponds to (theta, gain) for a resonator
    pub resonators: Vec<(f64, f64)>,
}

impl ScaledResonatorPlan {
    #[inline]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            resonators: Vec::with_capacity(size),
        }
    }

    #[inline]
    pub fn build_resonator_array(&self, sample_rate: f64) -> Result<ConjPoleResonatorArray, &'static str> {
        let mut res_array = ConjPoleResonatorArray::new(sample_rate, self.resonators.len());
        for peak in &self.resonators {
            res_array.add_resonator_theta(peak.0, 1.0, peak.1)?;
        }
        Ok(res_array)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ScaledResonatorPlanner {
    min_prominence: Option<f64>,
    min_threshold: Option<f64>,
    max_num_peaks: Option<usize>,
    /// value from 0.0 to 1.0 where 1.0 corresponds to sample rate
    min_freq: Option<f64>,
    /// value from 0.0 to 1.0 where 1.0 corresponds to sample rate
    max_freq: Option<f64>,
}

impl ScaledResonatorPlanner {
    /// Initialize a new resonator planner. Initializes all value to None.
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    #[inline]
    pub fn with_min_prominence(mut self, v: f64) -> Self {
        self.min_prominence = Some(v);
        self
    }

    #[inline]
    pub fn with_min_threshold(mut self, v: f64) -> Self {
        self.min_threshold = Some(v);
        self
    }

    #[inline]
    pub fn with_max_num_peaks(mut self, v: usize) -> Self {
        self.max_num_peaks = Some(v);
        self
    }

    #[inline]
    pub fn with_min_freq(mut self, v: f64) -> Self {
        self.min_freq = Some(v);
        self
    }

    #[inline]
    pub fn with_max_freq(mut self, v: f64) -> Self {
        self.max_freq = Some(v);
        self
    }

    fn slice_index_to_theta(log_spec_index: usize, min_bin: usize, spec_size: usize) -> f64 {
        (log_spec_index + min_bin) as f64 / spec_size as f64 * PI * 2.0
    }

    pub fn plan(&self, audio: &[f64], sample_rate: f64) -> ScaledResonatorPlan {
        if audio.len() > 3 {
            return ScaledResonatorPlan {
                resonators: vec![],
            }
        }
        let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
        let spec_size = 2_usize.pow(near_pow_2 as u32);
        let mut calc = FftCalculator::new(audio.len(), spec_size - audio.len()).unwrap(); // shouldn't fail
        
        let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
        let log_spectrum = spectrum.iter().map(|v| v.log10()).collect::<Vec<f64>>();

        let min_freq = self.min_freq.unwrap_or(0.0);
        let max_freq = self.max_freq.unwrap_or(1.0) * sample_rate;

        let min_bin = (min_freq / sample_rate * spec_size as f64 / 2.0).floor() as usize;
        let max_bin = (max_freq / sample_rate * spec_size as f64 / 2.0).floor() as usize;
        
        let spec_slice = &spectrum[min_bin..max_bin];
        let log_spec_slice = &log_spectrum[min_bin..max_bin];

        let mut spec_max = 0_f64;
        for v in spec_slice {
            if *v > spec_max {
                spec_max = *v;
            }
        }

        let mut log_spec_max = 0_f64;
        for v in log_spec_slice {
            if *v > log_spec_max {
                log_spec_max = *v;
            }
        }

        let mut peak_finder = PeakFinder::new(log_spec_slice);

        if let Some(v) = self.min_prominence {
            peak_finder.with_min_prominence(v);
        }
        if let Some(v) = self.min_threshold {
            peak_finder.with_min_height(v);
        }

        let mut peaks = peak_finder.find_peaks();

        peaks.sort_by(|a, b| b.height.partial_cmp(&a.height).unwrap());

        // number of resonators in output array
        let n = peaks.len().min(self.max_num_peaks.unwrap_or(100));
        
        let mut plan = ScaledResonatorPlan::with_capacity(n);

        for i in 0..n {
            let bin = peaks[i].middle_position();
            plan.resonators.push((Self::slice_index_to_theta(bin, min_bin, spec_size), spec_slice[bin] / spec_max));
        }

        plan
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use wav_util::*;
    use bode_plot::*;
    use bode_plot::plot::BodePlotTransferFunction;

    #[test]
    fn test_scaled_peak_planner() {
        let ([chan1, _chan2], sample_rate) = read_wave("./tests/fm.wav").unwrap();

        let plan = ScaledResonatorPlanner::new()
            .with_max_num_peaks(20)
            .plan(&chan1[..], sample_rate as f64);

        println!("{:?}", plan);
        let mut array = plan.build_resonator_array(sample_rate as f64).unwrap();

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
}