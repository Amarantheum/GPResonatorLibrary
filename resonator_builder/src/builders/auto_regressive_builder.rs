use auto_regressive::AutoRegressiveModel;

pub struct ARBuilder {
    pub model: AutoRegressiveModel,
    pub sample_rate: f64,
    pub min_freq: f64,
    pub max_freq: f64,
    pub max_num_peaks: usize,
}

