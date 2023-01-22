//! This crate contains abstractions for creating general purpose resonators.
//! A general purpose resonator is an array of resonators that can be used to create resonances at arbitrary frequencies for real-time audio.

pub mod resonator;

#[cfg(test)]
mod tests {
    use bode_plot::{create_log_plot, plot::BodePlotTransferFunction, DEFAULT_WIDTH, DEFAULT_HEIGHT}; 
    use num_complex::Complex;
    use std::error::Error;
    use crate::resonator::ConjPoleResonator;

    #[test]
    fn test_basic_bode_plot() {
        let test_transfer1 = ConjPoleResonator::new_polar(0.99999999, 3.14159 / 2.0, 1.0);
        let test_transfer2 = ConjPoleResonator::new_polar(0.99999, 3.14159 / 4.0, 1.0);
        let test_transfer3 = ConjPoleResonator::new_polar(0.99999999, 3.14159 / 4.0 * 3.0, 1.0);
        create_log_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_transfer1 as &dyn BodePlotTransferFunction, &test_transfer2 as &dyn BodePlotTransferFunction, &test_transfer3 as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
