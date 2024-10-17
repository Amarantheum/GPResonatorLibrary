//! This module contains implementations for resonator arrays.
pub use basic_resonator_array::BasicResonatorArray;
pub use phased_resonator_array::PhasedResonatorArray;

mod basic_resonator_array;
mod phased_resonator_array;

// used to keep track of the state of resonators to make process_buf treat new data as part of old data
#[derive(Debug, Clone)]
struct ConjPoleResonatorState {
    x_1: f64,
    y_1: f64,
    y_2: f64,
}

impl Default for ConjPoleResonatorState {
    fn default() -> Self {
        Self {
            x_1: 0_f64,
            y_1: 0_f64,
            y_2: 0_f64,
        }
    }
}