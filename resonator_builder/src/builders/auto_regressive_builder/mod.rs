use auto_regressive::AutoRegressiveModel;
use crate::plan_types::{PhasedResonatorPlan, PhasedResonatorArrayPlan};

mod rational_fn;

pub struct ARBuilder {
    order: usize,
}

impl ARBuilder {
    pub fn new(order: usize) -> Self {
        assert!(order % 2 == 0); // use even orders for now
        Self {
            order,
        }
    }

    // #[inline]
    // fn ar_model_to_resonator_array_plan(ar_model: &AutoRegressiveModel) -> ResonatorArrayPlan {
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

}