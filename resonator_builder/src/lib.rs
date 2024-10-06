//! This library contains functions for building a resonator array from audio data.
//! [`scaled_builder::ResonatorArrayPlanner`] can be found in the scaled_builder module.
pub mod fft;
pub mod builders;
mod plan_types;

pub use plan_types::resonator_plan::ResonatorPlan;
pub use plan_types::resonator_array_plan::ResonatorArrayPlan;
pub use plan_types::ResonatorBuilder;