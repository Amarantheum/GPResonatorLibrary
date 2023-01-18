use plotters::chart::ChartState;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::{backend::BGRXPixel};
use std::error::Error;
use crate::pixel_buffer::PixelBuffer;
use std::borrow::{BorrowMut};
use crate::colors::*;
use num_complex::{Complex, ComplexFloat};
use std::f64::consts::PI;

/// Type that stores data about a bode plot and is used to display bode plots in windows
#[allow(unused)]
pub(super) struct BodePlot {
    chart: ChartState<Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    pub pixel_buf: PixelBuffer,
    width: usize,
    height: usize,
}

impl BodePlot {
    fn new(width: usize, height: usize, path: Vec<(f64, f64)>, min_y: f64, max_y: f64) -> Result<Self, Box<dyn Error>> {
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
                .build_cartesian_2d(0.0..1_f64, min_y..max_y)?;
    
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
        Ok(
            Self {
                chart: cs,
                pixel_buf: buf,
                width,
                height,
            }
        )
    }
    /// This function constructs a chart state given a list of transfer functions.
    pub fn from_list(width: usize, height: usize, t_fns: Vec<&dyn BodePlotTransferFunction>) -> Result<Self, Box<dyn Error>> {
        // build line plot from given transfer functions
        let mut path = Vec::with_capacity(width + 1);
        let mut max = f64::MIN;
        for i in 0..=width {
            let arg = i as f64 / width as f64 * PI;
            let z = Complex::<f64>::from_polar(1_f64, arg);
            let mut cur = 0_f64;
            for tfn in &t_fns {
                cur += tfn.get_value(z).abs();
            }
            if cur > max {
                max = cur;
            }
            path.push((i as f64 / width as f64, cur));
        }
        Self::new(width, height, path, 0.0, max)
    }

    pub fn from_list_log(width: usize, height: usize, t_fns: Vec<&dyn BodePlotTransferFunction>) -> Result<Self, Box<dyn Error>> {
        // build line plot from given transfer functions
        let mut path = Vec::with_capacity(width + 1);
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        for i in 0..=width {
            let arg = i as f64 / width as f64 * PI;
            let z = Complex::<f64>::from_polar(1_f64, arg);
            let mut cur = 0_f64;
            for tfn in &t_fns {
                cur += tfn.get_value(z).abs().log10();
            }
            if cur > max {
                max = cur;
            }
            if cur < min {
                min = cur;
            }
            path.push((i as f64 / width as f64, cur));
        }
        Self::new(width, height, path, min.max(-8.0), max)
    }
}

/// Trait for objects that can be interpretted as a plotable transfer function
pub trait BodePlotTransferFunction {
    /// Takes a complex value on the Z plane and returns the output of the transfer function at that point
    /// # Arguments
    /// * `z` - A complex value in the Z plane
    fn get_value(&self, z: Complex<f64>) -> Complex<f64>;
}