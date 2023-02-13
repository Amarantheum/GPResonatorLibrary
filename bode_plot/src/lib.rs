//! A simple library for displaying bode plots in windows.
//! Implement [`crate::plot::BodePlotTransferFunction`] for your own type and pass into [`create_plot`] or [`create_log_plot`] to display the frequency response of the transfer function.

use plot::{BodePlot, BodePlotTransferFunction, LTISystem};
use minifb::{Key, Window, WindowOptions};
use plotters::coord::ranged1d::AsRangedCoord;
use std::error::Error;
use std::time::SystemTime;
use std::borrow::{Borrow};
use plotters::chart::ChartState;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::{backend::BGRXPixel};
use pixel_buffer::PixelBuffer;
use std::borrow::{BorrowMut};
use colors::*;
use std::ops::Range;

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

pub fn create_simulation_plot(name: String, width: usize, height: usize, systems: Vec<&dyn LTISystem>) -> Result<(), Box<dyn Error>> {
    let bode_plot = BodePlot::from_list_sim(width, height, systems)?;
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

fn basic_plot(width: usize, height: usize, path: Vec<(f64, f64)>, x_spec: Range<f64>, y_spec: Range<f64>) -> Result<(ChartState<Cartesian2d<RangedCoordf64, RangedCoordf64>>, PixelBuffer), Box<dyn Error>> {
    let mut buf = PixelBuffer::new(width, height);
    
    // begin constructing chart
    let cs = {
        let root =
            BitMapBackend::<BGRXPixel>::with_buffer_and_format(buf.borrow_mut(), (width as u32, height as u32))?
                .into_drawing_area();
        root.fill(&BLACK)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_all_label_area_size(30)
            .build_cartesian_2d(x_spec, y_spec)?;
        chart
            .configure_mesh()
            .label_style(("sans-serif", 15).into_font().color(&LIGHT_BLUE))
            .axis_style(&LIGHT_BLUE)
            .draw()?;
        
        chart.draw_series(vec![PathElement::new(path, &LIGHT_BLUE)]).unwrap();
        let cs = chart.into_chart_state();
        root.present()?;
        cs
    };
    Ok((cs, buf))
}


pub fn create_generic_plot(name: String, width: usize, height: usize, path: Vec<(f64, f64)>, x_spec: Range<f64>, y_spec: Range<f64>) -> Result<(), Box<dyn Error>> {
    let (_cs, buf) = basic_plot(width, height, path, x_spec, y_spec)?;
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
                window.update_with_buffer(buf.borrow(), width, height).unwrap();
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
