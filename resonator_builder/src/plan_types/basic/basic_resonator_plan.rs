use serde::{Serialize, Deserialize};
use gp_resonator::resonators::ConjPoleResonator;
use super::BasicResonatorPlanError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasicResonatorPlan {
    mag: f64,
    arg: f64,
    gain: f64,
}

impl BasicResonatorPlan {
    /// Create a new [`BasicResonatorPlan`] with the given magnitude, argument, and gain.
    /// # Arguments
    /// * `mag` - The magnitude of the resonator (must be in range [0, 1))
    /// * `arg` - The argument of the resonator (must be in range [0, π])
    /// * `gain` - The gain of the resonator
    #[inline]
    pub fn new(mag: f64, arg: f64, gain: f64) -> Self {
        assert!(mag < 1.0 && mag >= 0.0);
        assert!(arg >= 0.0 && arg <= std::f64::consts::PI);
        Self {
            mag,
            arg,
            gain,
        }
    }

    #[inline]
    pub fn get_conj_pole_resonator(&self) -> ConjPoleResonator {
        ConjPoleResonator::new_polar(self.mag, self.arg, self.gain)
    }
}

// getters and setters
impl BasicResonatorPlan {
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

    /// Get the gain of the resonator.
    #[inline]
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Set the magnitude of the resonator. The magnitude must be in the range [0, 1).
    #[inline]
    pub fn set_mag(&mut self, mag: f64) -> Result<(), BasicResonatorPlanError> {
        if mag < 0.0 || mag >= 1.0 {
            return Err(BasicResonatorPlanError::InvalidMagnitude(mag));
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
    pub fn set_arg(&mut self, arg: f64) -> Result<(), BasicResonatorPlanError> {
        if arg < 0.0 || arg > std::f64::consts::PI {
            return Err(BasicResonatorPlanError::InvalidArgument(arg));
        }
        self.arg = arg;
        Ok(())
    }

    /// Set the argument of the resonator without checking bounds.
    #[inline]
    pub fn set_arg_unchecked(&mut self, arg: f64) {
        self.arg = arg;
    }

    /// Set the gain of the resonator.
    #[inline]
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }
}