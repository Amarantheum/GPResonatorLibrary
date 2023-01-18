mod resonator;

#[cfg(test)]
mod tests {
    use bode_plot::{create_plot, plot::BodePlotTransferFunction, DEFAULT_WIDTH, DEFAULT_HEIGHT}; 
    use num_complex::Complex;
    use std::error::Error;
    use crate::resonator::ConjPoleResonator;

    #[test]
    fn test_basic_bode_plot() {
        let test_transfer = ConjPoleResonator::new_polar(0.999, 3.14159 / 2.0, 1.0);
        create_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_transfer as &dyn BodePlotTransferFunction]).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
