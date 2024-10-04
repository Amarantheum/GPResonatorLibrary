pub mod resonator_plan;
pub mod resonator_array_plan;

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