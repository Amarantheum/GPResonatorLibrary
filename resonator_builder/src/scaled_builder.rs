//! Implementation of the [`ScaledResonatorPlanner`] used for building resonators.
//! This method builds a resonator array by setting a uniform decay rate for all resonators and scaling each resonator's gain to match the amplitude of the corresponding peak in the spectrum.
use std::{f64::consts::PI};
use crate::fft::{FftCalculator, window::{Rectangular, WindowFunction}};
use find_peaks::{PeakFinder};
use gp_resonator::resonator_array::ConjPoleResonatorArray;
use serde::{Serialize, Deserialize};

/// A type representing a plan for building a resonator array using the scaled method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaledResonatorPlan {
    /// Each value corresponds to (theta, gain) for a resonator
    pub resonators: Vec<(f64, f64)>,
}

impl ScaledResonatorPlan {
    /// Initialize an empty resonator plan with given capacity.
    #[inline]
    pub fn with_capacity(size: usize) -> Self {
        Self {
            resonators: Vec::with_capacity(size),
        }
    }

    /// Build a resonator array from this plan.
    #[inline]
    pub fn build_resonator_array(&self, sample_rate: f64) -> Result<ConjPoleResonatorArray, &'static str> {
        let mut res_array = ConjPoleResonatorArray::new(sample_rate, self.resonators.len());
        for peak in &self.resonators {
            res_array.add_resonator_theta(peak.0, 1.0, peak.1)?;
        }
        Ok(res_array)
    }

    /// Obtain an iterator over the resonators in this plan.
    pub fn iter(&self) -> std::slice::Iter<(f64, f64)> {
        self.resonators.iter()
    }

    /// Initialize an empty resonator plan.
    #[inline]
    pub fn empty() -> Self {
        Self {
            resonators: vec![],
        }
    }

    #[inline]
    pub fn sort(&mut self) {
        self.resonators.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    #[inline]
    pub fn normalize(&mut self) {
        let mut gain_sum = 0.0;
        for (_, gain) in &self.resonators {
            gain_sum += gain;
        }
        println!("gain_sum: {}", gain_sum);
        for (_, gain) in &mut self.resonators {
            *gain /= gain_sum;
        }
    }
}

impl IntoIterator for ScaledResonatorPlan {
    type Item = (f64, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.resonators.into_iter()
    }
}

/// A type used to plan the creation of a resonator array using the scaled method.
/// Uses a builder pattern to set the parameters for the planner.
#[derive(Default, Clone, Copy, Debug)]
pub struct ScaledResonatorPlanner {
    /// value that filters peaks based on topographic prominence
    min_prominence: Option<f64>,
    /// value that filters peaks based on height
    min_threshold: Option<f64>,
    /// the number of peaks to find
    max_num_peaks: Option<usize>,
    /// value from 0.0 to 1.0 where 1.0 corresponds to sample rate
    min_freq: Option<f64>,
    /// value from 0.0 to 1.0 where 1.0 corresponds to sample rate
    max_freq: Option<f64>,
}

impl ScaledResonatorPlanner {
    /// Initialize a new empty resonator planner. Initializes all values to None.
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the prominence threshold for peaks
    #[inline]
    pub fn with_min_prominence(mut self, v: f64) -> Self {
        self.min_prominence = Some(v);
        self
    }

    /// Set the minimum height for peaks
    #[inline]
    pub fn with_min_threshold(mut self, v: f64) -> Self {
        self.min_threshold = Some(v);
        self
    }

    /// Set the maximum number of peaks
    #[inline]
    pub fn with_max_num_peaks(mut self, v: usize) -> Self {
        self.max_num_peaks = Some(v);
        self
    }

    /// Set the lower bound for frequency when searching for peaks
    #[inline]
    pub fn with_min_freq(mut self, v: f64) -> Self {
        self.min_freq = Some(v);
        self
    }

    /// Set the upper bound for frequency when searching for peaks
    #[inline]
    pub fn with_max_freq(mut self, v: f64) -> Self {
        self.max_freq = Some(v);
        self
    }

    // Get the angle corresponding to a bin index
    fn slice_index_to_theta(log_spec_index: usize, min_bin: usize, spec_size: usize) -> f64 {
        (log_spec_index + min_bin) as f64 / spec_size as f64 * PI * 2.0
    }

    /// Perform the calculations with the planner to create a [`ScaledResonatorPlan`].
    pub fn plan(&self, audio: &[f64]) -> ScaledResonatorPlan {
        if audio.len() < 3 {
            return ScaledResonatorPlan {
                resonators: vec![],
            }
        }
        let near_pow_2 = ((audio.len() - 1).ilog2() + 1) as usize;
        let spec_size = 2_usize.pow(near_pow_2 as u32);
        let mut calc = FftCalculator::new(audio.len(), spec_size - audio.len()).unwrap(); // shouldn't fail
        
        let spectrum = calc.real_fft(audio, Rectangular::real_window).into_iter().map(|v| v.norm()).collect::<Vec<f64>>();
        let log_spectrum = spectrum.iter().map(|v| v.log10()).collect::<Vec<f64>>();

        let min_bin = (self.min_freq.unwrap_or(0.0) * spec_size as f64 / 2.0).floor() as usize;
        let max_bin = (self.max_freq.unwrap_or(1.0) * spec_size as f64 / 2.0).floor() as usize;
        
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
    use rand::random;
    use wav_util::*;
    use bode_plot::*;
    use bode_plot::plot::BodePlotTransferFunction;

    #[test]
    fn test_scaled_peak_planner() {
        let ([chan1, _chan2], sample_rate) = read_wave("./tests/fm.wav").unwrap();

        let plan = ScaledResonatorPlanner::new()
            .with_max_num_peaks(20)
            .with_min_freq(0.0)
            .with_min_prominence(1.0)
            .plan(&chan1[..]);

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

    #[test]
    fn test_scaled_peak_planner_buffer() {
        let ([chan1, _chan2], sample_rate) = read_wave("./tests/fm.wav").unwrap();

        let plan = ScaledResonatorPlanner::new()
            .with_max_num_peaks(20)
            .with_min_freq(0.0)
            .with_min_prominence(1.0)
            .plan(&chan1[..]);

        let mut array = plan.build_resonator_array(sample_rate as f64).unwrap();

        let ([chan1, chan2], _) = read_wave("./tests/test_noise.wav").unwrap();
        let mut out_chan1 = vec![0_f64; chan1.len()];
        let mut out_chan2 = out_chan1.clone();

        let start = std::time::Instant::now();
        let step_size = chan1.len() / 2;
        for i in 0..1 {
            array.process_buf(&chan1[i * step_size..(i + 1) * step_size], &mut out_chan1[i * step_size..(i + 1) * step_size]);
        }
        array.process_buf(&chan1[1 * step_size..], &mut out_chan1[1 * step_size..]);
        println!("Took {}s to process signal", start.elapsed().as_secs_f64());
        array.reset_state();
        for i in 0..1 {
            array.process_buf(&chan2[i * step_size..(i + 1) * step_size], &mut out_chan2[i * step_size..(i + 1) * step_size]);
        }
        array.process_buf(&chan2[1 * step_size..], &mut out_chan2[1 * step_size..]);
        //array.process_buf(&chan2[..], &mut out_chan2[..]);

        write_wave([out_chan1, out_chan2], "./tests/test_fm_resonator.wav", 48_000).unwrap();
        create_plot("Harmonincs".into(), DEFAULT_WIDTH * 2, DEFAULT_HEIGHT, vec![&array as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_noise() {
        let num_iter = 1000;
        let len = 64 * num_iter;
        let plan = ScaledResonatorPlan {
            resonators: vec![(PI / 512.0, 0.999)],
        };

        let mut array = plan.build_resonator_array(48_000.0).unwrap();
        let mut noise: Vec<f64> = Vec::with_capacity(len);
        for _ in 0..len {
            noise.push(random::<f64>() * 2.0 - 1.0);
        }

        let mut out: Vec<f64> = vec![0.0; len];
        
        for i in 0..num_iter {
            array.process_buf(&noise[i * 64..(i + 1) * 64], &mut out[i * 64..(i + 1) * 64]);
        }

        let in_path = noise.iter().enumerate().map(|(i, v)| (i as f64, *v)).collect::<Vec<(f64, f64)>>();
        let out_path = out.iter().enumerate().map(|(i, v)| (i as f64, *v)).collect::<Vec<(f64, f64)>>();
        create_generic_plot("Noise Input".into(), 1920, 1080, in_path, 0.0..(len as f64), -1.0..1.0).unwrap();
        create_generic_plot("Noise Output".into(), 1920, 1080, out_path, 0.0..(len as f64), -100.0..100.0).unwrap();
        loop{}
    }

    #[test]
    fn test_sort() {
        let mut plan = ScaledResonatorPlan {
            resonators: vec![(0.0, 0.0), (0.5, 0.0), (0.25, 0.0)],
        };
        plan.sort();
        assert_eq!(plan.resonators, vec![(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]);
    }

    #[test]
    fn test_impulse_response() {
        let mut builder = ScaledResonatorPlan {
            resonators: vec![
                (
                    0.03343598748594483,
                    1.0,
                ),
                (
                    0.01712545738975454,
                    0.7161415539141641,
                ),
                (
                    0.07100653256423768,
                    0.690909813356805,
                ),
                (
                    0.02629338944235232,
                    0.5486664261695459,
                ),
                (
                    0.020385166564011528,
                    0.45759929361676804,
                ),
                (
                    0.051400340619074336,
                    0.34739678897313697,
                ),
                (
                    0.13895708777760946,
                    0.25440234160011005,
                ),
                (
                    0.008520783907708522,
                    0.25300848182575175,
                ),
                (
                    0.042280345466097984,
                    0.2288002965861071,
                ),
                (
                    0.09233845289577239,
                    0.2209617787032956,
                ),
                (
                    0.04474909579660144,
                    0.19591370164261154,
                ),
                (
                    0.08285893099563534,
                    0.17808750319786543,
                ),
                (
                    0.1004757416065095,
                    0.14233475089700245,
                ),
                (
                    0.16334498546001008,
                    0.11090566246945736,
                ),
                (
                    0.0772383295150231,
                    0.108647778620586,
                ),
                (
                    0.1154320542883945,
                    0.09817091748367965,
                ),
                (
                    0.18816431523900354,
                    0.06081983802961128,
                ),
                (
                    0.23777900634717974,
                    0.030778812199537618,
                ),
                (
                    0.1768272384785362,
                    0.017708051270270503,
                ),
                (
                    0.20869329250187935,
                    0.008833948746736358,
                ),
                (
                    0.2627061941503214,
                    0.008230914261496539,
                ),
                (
                    0.28805282982515057,
                    0.00537908691131123,
                ),
                (
                    0.313207717901494,
                    0.003485969651554882,
                ),
                (
                    0.34653584736329063,
                    0.0023440166171421906,
                ),
                (
                    0.3366967987159929,
                    0.0021921644503870106,
                ),
                (
                    0.3612764439968792,
                    0.0020896946988739557,
                ),
                (
                    0.4397371644522487,
                    0.0013868248931222463,
                ),
                (
                    0.37257756808263043,
                    0.0013043650356380752,
                ),
                (
                    0.4853850771167519,
                    0.0010744871111628993,
                ),
                (
                    0.3861197422256834,
                    0.0010452333532522716,
                ),
                (
                    0.4036766317120308,
                    0.0010388356961683572,
                ),
                (
                    0.5029419666030992,
                    0.001029455078998618,
                ),
                (
                    0.4568146849423818,
                    0.0008544032837893364,
                ),
                (
                    0.5559481933594913,
                    0.0007599276817099142,
                ),
                (
                    0.4682116828273759,
                    0.0006610675198007496,
                ),
                (
                    0.5460012866880454,
                    0.0005662866418615921,
                ),
                (
                    0.5705929161938371,
                    0.0005051843176634989,
                ),
                (
                    0.5618803846876429,
                    0.00039002276575829837,
                ),
                (
                    0.641371748484873,
                    0.00016634736176602638,
                ),
                (
                    0.6276378267433343,
                    0.00015190687784542392,
                ),
                (
                    0.6624879527681112,
                    0.00013257683182968174,
                ),
                (
                    0.6923286727824491,
                    0.00010635903936822515,
                ),
                (
                    0.7192572261447854,
                    6.643912448642534e-5,
                ),
                (
                    0.7747801401313024,
                    5.2006818904724064e-5,
                ),
                (
                    0.736286809735297,
                    5.1451644383140826e-5,
                ),
                (
                    0.7463655428807019,
                    3.938215675989191e-5,
                ),
            ],
        };
        builder.sort();
        builder.normalize();
        let mut resonator = builder.build_resonator_array(48_000.0)
            .unwrap();
        let mut input = vec![0_f64; 1000];
        // for i in 0..input.len() {
        //     input[i] = random::<f64>() * 2.0 - 1.0;
        // }
        input[0] = 1.0;
        let mut out = vec![0_f64; input.len()];
        resonator.process_buf(&input[..], &mut out[..]);
        println!("{:?}", out);
        println!("max_value: {:?}", out.iter().fold(0_f64, |acc, v| acc.max(*v)));
    }
}