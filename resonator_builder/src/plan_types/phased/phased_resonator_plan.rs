use super::PhasedResonatorPlanError;
use serde::{Serialize, Deserialize};
use gp_resonator::resonators::PhasedConjPoleResonator;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseGain {
    phase: f64,
    gain: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XConsts {
    x_0_const: f64,
    x_1_const: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhaseInfo {
    PhaseGain(PhaseGain),
    XConsts(XConsts),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhasedResonatorPlan {
    mag: f64,
    arg: f64,
    phase_info: PhaseInfo,
}

impl PhasedResonatorPlan {
    /// Create a new [`PhasedResonatorPlan`] given a complex number in polar form and a phase.
    /// The poles will be at the location of the complex number and its conjugate.
    /// # Arguments
    /// * `mag` - The magnitude of the complex number (must be in range [0, 1))
    /// * `arg` - The argument of the commplex number (must be in range [0, π])
    /// * `phase` - The phase of the resonator
    /// * `gain` - The gain of the filter
    #[inline]
    pub fn new_with_phase(mag: f64, arg: f64, phase: f64, gain: f64) -> Self {
        debug_assert!(mag < 1.0 && mag >= 0.0);
        debug_assert!(arg >= 0.0 && arg <= std::f64::consts::PI);
        debug_assert!(phase >= 0.0 && phase <= std::f64::consts::PI * 2.0);
        Self {
            mag,
            arg,
            phase_info: PhaseInfo::PhaseGain(PhaseGain { phase, gain }),
        }
    }

    /// Create a new [`PhasedResonatorPlan`] given a complex number in polar form and constants for x_0 and x_1.
    /// x_0_const and x_1_const are used in the difference equation for the resonator (x_0_const * x[n] + x_1_const * x[n-1]).
    /// The poles will be at the location of the complex number and its conjugate.
    /// # Arguments
    /// * `mag` - The magnitude of the complex number (must be in range [0, 1))
    /// * `arg` - The argument of the commplex number (must be in range [0, π])
    /// * `x_0_const` - The constant for x_0
    #[inline]
    pub fn new_with_consts(mag: f64, arg: f64, x_0_const: f64, x_1_const: f64) -> Self {
        debug_assert!(mag < 1.0 && mag >= 0.0);
        debug_assert!(arg >= 0.0 && arg <= std::f64::consts::PI);
        Self {
            mag,
            arg,
            phase_info: PhaseInfo::XConsts(XConsts { x_0_const: x_0_const, x_1_const: x_1_const }),
        }
    }

    #[inline]
    pub fn get_phased_resonator(&self) -> PhasedConjPoleResonator {
        match &self.phase_info {
            PhaseInfo::PhaseGain(phase_gain) => PhasedConjPoleResonator::new_with_phase(self.mag, self.arg, phase_gain.phase, phase_gain.gain),
            PhaseInfo::XConsts(x_consts) => PhasedConjPoleResonator::new_with_consts(self.mag, self.arg, x_consts.x_0_const, x_consts.x_1_const),
        }
    }
}

impl PhasedResonatorPlan {

    /// Get the magnitude of the resonator.
    #[inline]
    pub fn mag(&self) -> f64 {
        self.mag
    }

    /// Get the argument of the resonator.
    #[inline]
    pub fn arg(&self) -> f64 {
        self.arg
    }

    /// Set the magnitude of the resonator. The magnitude must be in the range [0, 1).
    #[inline]
    pub fn set_mag(&mut self, mag: f64) -> Result<(), PhasedResonatorPlanError> {
        if mag < 0.0 || mag >= 1.0 {
            return Err(PhasedResonatorPlanError::InvalidMagnitude(mag));
        }
        self.mag = mag;
        Ok(())
    }

    /// Set the magnitude of the resonator without checking bounds
    #[inline]
    pub fn set_mag_unchecked(&mut self, mag: f64) {
        self.mag = mag;
    }

    /// Set the argument of the resonator. The argument must be in the range [0, π].
    #[inline]
    pub fn set_arg(&mut self, arg: f64) -> Result<(), PhasedResonatorPlanError> {
        if arg < 0.0 || arg > std::f64::consts::PI {
            return Err(PhasedResonatorPlanError::InvalidArgument(arg));
        }
        self.arg = arg;
        Ok(())
    }

    /// Set the argument of the resonator without checking bounds.
    #[inline]
    pub fn set_arg_unchecked(&mut self, arg: f64) {
        self.arg = arg;
    }
}