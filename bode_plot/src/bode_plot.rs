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
pub struct BodePlot {
    chart: ChartState<Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    pub pixel_buf: PixelBuffer,
    width: usize,
    height: usize,
}

impl BodePlot {
    /// This function constructs a chart state given a list of transfer functions.
    #[allow(unused)]
    pub fn from_list(width: usize, height: usize, t_fns: Vec<Box<&dyn BodePlotTransferFunction>>) -> Result<Self, Box<dyn Error>> {
        let mut buf = PixelBuffer::new(width, height);
        
        // begin constructing chart
        let cs = {
            let root =
                BitMapBackend::<BGRXPixel>::with_buffer_and_format(buf.borrow_mut(), (width as u32, height as u32))?
                    .into_drawing_area();
            root.fill(&BLACK)?;
    
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

            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .set_all_label_area_size(30)
                .build_cartesian_2d(0.0..1_f64, 0.0..max)?;
    
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
}

pub trait BodePlotTransferFunction {
    fn get_value(&self, z: Complex<f64>) -> Complex<f64>;
}