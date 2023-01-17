use bode_plot::{BodePlot, BodePlotTransferFunction};
use minifb::{Key, Window, WindowOptions};
use plotters::prelude::*;
use plotters::{backend::BGRXPixel};
use std::error::Error;
use std::time::SystemTime;
use pixel_buffer::PixelBuffer;
use std::borrow::{Borrow, BorrowMut};
use colors::*;

mod pixel_buffer;
mod bode_plot;
mod colors;

// starter code from https://github.com/plotters-rs/plotters-minifb-demo
const W: usize = 800;
const H: usize = 600;

const FRAME_RATE: f64 = 60.0;

fn get_window_title(fx: f64, fy: f64, iphase: f64) -> String {
    format!(
        "x={:.1}Hz, y={:.1}Hz, phase={:.1} +/-=Adjust y 9/0=Adjust x <Esc>=Exit",
        fx, fy, iphase
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    // type for testing plotting function
    pub struct RealZeroTransferFunction(pub f64);

    impl BodePlotTransferFunction for RealZeroTransferFunction {
        fn get_value(&self, z: Complex<f64>) -> Complex<f64> {
            z - Complex::<f64>::new(self.0, 0.0)
        }
    }

    fn create_basic_low_pass() -> Result<(), Box<dyn Error>> {
        let mut window = Window::new(
            "BODE PLOT",
            W,
            H,
            WindowOptions::default(),
        )?;
    
        let test_transfer = RealZeroTransferFunction(-1.0);
        let bode_plot = BodePlot::from_list(W, H, vec![Box::new(&test_transfer as &dyn BodePlotTransferFunction)])?;
    
        let start_ts = SystemTime::now();
        let mut last_flushed = 0.0;
    
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let epoch = SystemTime::now()
                .duration_since(start_ts)
                .unwrap()
                .as_secs_f64();
    
            if epoch - last_flushed > 1.0 / FRAME_RATE {
                window.update_with_buffer(bode_plot.pixel_buf.borrow(), W, H)?;
                last_flushed = epoch;
            }
        }
        Ok(())
    }

    #[test]
    fn test_basic_low_pass() {
        create_basic_low_pass().unwrap();
    }
}
