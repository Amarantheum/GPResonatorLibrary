use plot::{BodePlot, BodePlotTransferFunction};
use minifb::{Key, Window, WindowOptions};
use std::error::Error;
use std::time::SystemTime;
use std::borrow::{Borrow};

mod pixel_buffer;
pub mod plot;
mod colors;

// starter code from https://github.com/plotters-rs/plotters-minifb-demo
/// Default window width in pixels
pub const DEFAULT_WIDTH: usize = 1000;
/// Default window height in pixels
pub const DEFAULT_HEIGHT: usize = 500;

const FRAME_RATE: f64 = 60.0;

/// Asynchronously opens a window displaying the bode plot of the given transfer function with linear scale.
/// # Arguments
/// * `name` - The name displayed on the window
/// * `width` - The width of the window (can use [`crate::DEFAULT_WIDTH`])
/// * `height` - The height of the window (can use [`crate::DEFAULT_HEIGHT`])
/// * `t_fns` - A list of trait objects used to generate the bode plot
pub fn create_plot(name: String, width: usize, height: usize, t_fns: Vec<&dyn BodePlotTransferFunction>) -> Result<(), Box<dyn Error>> {
    let bode_plot = BodePlot::from_list(width, height, t_fns)?;
    create_plot_backend(name, width, height, bode_plot)
}

/// Asynchronously opens a window displaying the bode plot of the given transfer function with logarithmic scale.
/// # Arguments
/// * `name` - The name displayed on the window
/// * `width` - The width of the window (can use [`crate::DEFAULT_WIDTH`])
/// * `height` - The height of the window (can use [`crate::DEFAULT_HEIGHT`])
/// * `t_fns` - A list of trait objects used to generate the bode plot
pub fn create_log_plot(name: String, width: usize, height: usize, t_fns: Vec<&dyn BodePlotTransferFunction>) -> Result<(), Box<dyn Error>> {
    let bode_plot = BodePlot::from_list_log(width, height, t_fns)?;
    create_plot_backend(name, width, height, bode_plot)
}

fn create_plot_backend(name: String, width: usize, height: usize, bode_plot: BodePlot) -> Result<(), Box<dyn Error>> {
    let start_ts = SystemTime::now();
    let mut last_flushed = 0.0;
    std::thread::spawn(move || {
        let mut window = Window::new(
            name.as_str(),
            width,
            height,
            WindowOptions::default(),
        ).unwrap();
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let epoch = SystemTime::now()
                .duration_since(start_ts)
                .unwrap()
                .as_secs_f64();
    
            if epoch - last_flushed > 1.0 / FRAME_RATE {
                window.update_with_buffer(bode_plot.pixel_buf.borrow(), width, height).unwrap();
                last_flushed = epoch;
            }
        }
    });
    
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
        let test_transfer = RealZeroTransferFunction(0.5);
        create_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&test_transfer as &dyn BodePlotTransferFunction])?;
        Ok(())
    }
    fn create_band_pass() -> Result<(), Box<dyn Error>> {
        create_log_plot("TEST PLOT".into(), DEFAULT_WIDTH, DEFAULT_HEIGHT, vec![&RealZeroTransferFunction(1.0) as &dyn BodePlotTransferFunction, &RealZeroTransferFunction(-1.0) as &dyn BodePlotTransferFunction])?;
        Ok(())
    }

    #[test]
    fn test_basic_filters() {
        create_basic_low_pass().unwrap();
        create_band_pass().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
