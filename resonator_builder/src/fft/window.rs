use std::f64::consts::PI;

pub type RealWindowFn = fn(&mut [f64]);

pub fn null_window_fn(_buf: &mut [f64]) {}

pub trait WindowFunction {
    // applies window function to the buffer
    fn real_window(buffer: &mut [f64]);
}

pub struct BlackmanHarris;

impl WindowFunction for BlackmanHarris {
    fn real_window(buffer: &mut [f64]) {
        let size = buffer.len() as f64;
        for i in 0..buffer.len() {
            buffer[i] = buffer[i] * (
                0.35875 
                - 0.48829 * ((2 * i) as f64 * PI / size).cos() 
                + 0.14128 * ((4 * i) as f64 * PI / size).cos() 
                - 0.01168 * ((6 * i) as f64 * PI / size).cos()
            );
        }
    }
}

pub struct Rectangular;

impl WindowFunction for Rectangular {
    fn real_window(_buffer: &mut [f64]) {
        ()
    }
}