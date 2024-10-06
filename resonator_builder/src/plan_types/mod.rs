pub mod resonator_plan;
pub mod resonator_array_plan;

/// A type representing out of bounds errors for resonator plans.
/// The magnitude must be in the range [0, 1).
/// The argument must be in the range [0, π].
pub enum ResonatorPlanError {
    InvalidMagnitude(f64),
    InvalidArgument(f64),
}

impl ResonatorPlanError {
    pub fn as_str(&self) -> String {
        match self {
            ResonatorPlanError::InvalidMagnitude(s) => format!("Invalid magnitude: {} not in range [0, 1)", s),
            ResonatorPlanError::InvalidArgument(s) => format!("Invalid argument: {} not in range [0, π]", s),
        }
    }
}

impl std::fmt::Display for ResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for ResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::error::Error for ResonatorPlanError {}

/// A trait for types that can plan a resonator array.
pub trait ResonatorBuilder {
    /// Plan the resonator array based on the given audio.
    fn plan(&self, audio: &[f64], sample_rate: f64) -> resonator_array_plan::ResonatorArrayPlan;
}