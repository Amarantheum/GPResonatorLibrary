//! This module contains useful abstractions for resonating filters.
//! You are probably looking for [`ConjPoleResonator`].

mod basic_resonator;
mod phased_resonator;

pub use basic_resonator::ConjPoleResonator;
pub use phased_resonator::PhasedConjPoleResonator;