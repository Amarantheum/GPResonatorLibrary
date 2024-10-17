use serde::{Serialize, Deserialize};
use gp_resonator::resonator_arrays::BasicResonatorArray;
use super::basic_resonator_plan::BasicResonatorPlan;
use super::BasicResonatorPlanError;

/// A type representing a plan for building a resonator array using the scaled method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicResonatorArrayPlan {
    /// Each value corresponds to (theta, gain) for a resonator
    pub resonators: Vec<BasicResonatorPlan>,
    pub sample_rate: f64,
}

impl BasicResonatorArrayPlan {
    /// Initialize an empty resonator plan.
    /// # Arguments
    /// * `sample_rate` - The sample rate of the resonator plan
    #[inline]
    pub fn new(sample_rate: f64) -> Self {
        Self {
            resonators: vec![],
            sample_rate,
        }
    }

    /// Initialize an empty resonator plan with given capacity.
    /// # Arguments
    /// * `size` - The initial capacity of the resonator plan
    /// * `sample_rate` - The sample rate of the resonator plan
    #[inline]
    pub fn with_capacity(size: usize, sample_rate: f64) -> Self {
        Self {
            resonators: Vec::with_capacity(size),
            sample_rate,
        }
    }

    /// Build a resonator array from this plan.
    #[inline]
    pub fn build_resonator_array(&self) -> Result<BasicResonatorArray, &'static str> {
        let mut res_array = BasicResonatorArray::new(self.sample_rate, self.resonators.len());
        for peak in &self.resonators {
            res_array.add_resonator_raw(peak.get_conj_pole_resonator());
        }
        Ok(res_array)
    }

    /// Obtain an iterator over the resonators in this plan.
    pub fn iter(&self) -> std::slice::Iter<BasicResonatorPlan> {
        self.resonators.iter()
    }

    /// Initialize an empty resonator plan.
    #[inline]
    pub fn empty(sample_rate: f64) -> Self {
        Self {
            resonators: vec![],
            sample_rate,
        }
    }

    /// Sort the resonators in this plan from lowest to highest frequency.
    #[inline]
    pub fn sort(&mut self) {
        self.resonators.sort_by(|a, b| a.arg().partial_cmp(&b.arg()).unwrap());
    }

    /// Resample the resonator array plan with a new sample rate.
    /// Returns a [`ResampleResonatorArrayResult`] containing the new plan and a list of errors that occurred during resampling.
    /// Errors occur when any resonator has an argument out of range [0, π] which can occur when downsampling.
    /// If any resonators have an argument out of range [0, π], they will be omitted from the new plan unless `unwrap_allow_errors` is used.
    #[inline]
    pub fn resample(&self, new_sample_rate: f64) -> ResampleResonatorArrayResult {
        if new_sample_rate <= 0.0 {
            panic!("Sample rate must be greater than 0 to recalculate. Got {}", new_sample_rate);
        }

        let ratio = self.sample_rate / new_sample_rate;
        let mut new_resonator_plan = Self::with_capacity(self.resonators.len(), new_sample_rate);
        let mut errors = vec![];

        for resonator in &self.resonators {
            let mut new_resonator = resonator.clone();
            if let Err(e) = new_resonator.set_arg(resonator.arg() * ratio) {
                errors.push((new_resonator, e));
            } else {
                new_resonator_plan.resonators.push(new_resonator);
            }
        }

        ResampleResonatorArrayResult::new(new_resonator_plan, errors)
    }
}

impl IntoIterator for BasicResonatorArrayPlan {
    type Item = BasicResonatorPlan;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.resonators.into_iter()
    }
}

/// A type representing the result of resampling a resonator array plan with a new sample rate.
/// Contains the new plan and a list of errors that occurred during resampling (e.g. arg out of range (0, π)).
/// To ignore errors, use `unwrap_unchecked()`.
/// To get the a list of errors without destroying the object, use `errors()`.
#[derive(Debug)]
pub struct ResampleResonatorArrayResult {
    new_plan: BasicResonatorArrayPlan,
    errors: Vec<(BasicResonatorPlan, BasicResonatorPlanError)>,
}

impl ResampleResonatorArrayResult {
    fn new(new_plan: BasicResonatorArrayPlan, errors: Vec<(BasicResonatorPlan, BasicResonatorPlanError)>) -> Self {
        Self {
            new_plan,
            errors,
        }
    }

    /// Unwrap the result without checking for errors.
    pub fn unwrap_unchecked(self) -> BasicResonatorArrayPlan {
        self.new_plan
    }

    /// Get the list of errors that occurred during resampling.
    pub fn errors(&self) -> &Vec<(BasicResonatorPlan, BasicResonatorPlanError)> {
        &self.errors
    }

    /// Check if there are any errors in the result.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Unwrap the result, adding any resonators with args out of range [0, π] to the new plan.
    /// Only use this if you are sure that you want resonators with args out of range [0, π].
    /// Otherwise check for errors with `has_errors()` and handle them accordingly.
    pub fn unwrap_allow_errors(self) -> BasicResonatorArrayPlan {
        let mut plan = self.new_plan;
        let errors = self.errors;
        for (mut resonator, error) in errors {
            let new_arg = match error {
                BasicResonatorPlanError::InvalidArgument(arg) => {
                    arg
                },
                _ => panic!("There is a bug in the code if this shows up."),
            };
            resonator.set_arg_unchecked(new_arg);
            plan.resonators.push(resonator);
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort() {
        let mut plan = BasicResonatorArrayPlan::new(48_000.0);
        plan.resonators.push(BasicResonatorPlan::new(0.9, 0.0, 0.0));
        plan.resonators.push(BasicResonatorPlan::new(0.9, 0.5, 0.0));
        plan.resonators.push(BasicResonatorPlan::new(0.9, 0.25, 0.0));
        plan.sort();
        let mut expected = Vec::new();
        expected.push(BasicResonatorPlan::new(0.9, 0.0, 0.0));
        expected.push(BasicResonatorPlan::new(0.9, 0.25, 0.0));
        expected.push(BasicResonatorPlan::new(0.9, 0.5, 0.0));
        assert_eq!(plan.resonators, expected);
    }
}