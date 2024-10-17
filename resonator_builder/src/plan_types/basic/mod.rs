mod basic_resonator_plan;
mod basic_resonator_array_plan;

pub use basic_resonator_plan::BasicResonatorPlan;
pub use basic_resonator_array_plan::BasicResonatorArrayPlan;

/// A type representing out of bounds errors for resonator plans.
/// The magnitude must be in the range [0, 1).
/// The argument must be in the range [0, π].
pub enum BasicResonatorPlanError {
    InvalidMagnitude(f64),
    InvalidArgument(f64),
}

impl BasicResonatorPlanError {
    pub fn as_str(&self) -> String {
        match self {
            BasicResonatorPlanError::InvalidMagnitude(s) => format!("Invalid magnitude: {} not in range [0, 1)", s),
            BasicResonatorPlanError::InvalidArgument(s) => format!("Invalid argument: {} not in range [0, π]", s),
        }
    }
}

impl std::fmt::Display for BasicResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for BasicResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::error::Error for BasicResonatorPlanError {}

/// A trait for types that can plan a resonator array.
pub trait BasicResonatorBuilder {
    /// Plan the resonator array based on the given audio.
    fn plan(&self, audio: &[f64], sample_rate: f64) -> BasicResonatorArrayPlan;
}