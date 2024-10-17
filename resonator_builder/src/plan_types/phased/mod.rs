mod phased_resonator_plan;
mod phased_resonator_array_plan;

pub use phased_resonator_plan::PhasedResonatorPlan;
pub use phased_resonator_array_plan::PhasedResonatorArrayPlan;

/// A type representing out of bounds errors for resonator plans.
/// The magnitude must be in the range [0, 1).
/// The argument must be in the range [0, π].
pub enum PhasedResonatorPlanError {
    InvalidMagnitude(f64),
    InvalidArgument(f64),
    InvalidPhase(f64),
}

impl PhasedResonatorPlanError {
    pub fn as_str(&self) -> String {
        match self {
            PhasedResonatorPlanError::InvalidMagnitude(s) => format!("Invalid magnitude: {} not in range [0, 1)", s),
            PhasedResonatorPlanError::InvalidArgument(s) => format!("Invalid argument: {} not in range [0, π]", s),
            PhasedResonatorPlanError::InvalidPhase(s) => format!("Invalid phase: {} not in range [0, 2π)", s),
        }
    }
}

impl std::fmt::Display for PhasedResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for PhasedResonatorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::error::Error for PhasedResonatorPlanError {}

/// A trait for types that can plan a resonator array.
pub trait PhasedResonatorBuilder {
    /// Plan the resonator array based on the given audio.
    fn plan(&self, audio: &[f64], sample_rate: f64) -> PhasedResonatorArrayPlan;
}