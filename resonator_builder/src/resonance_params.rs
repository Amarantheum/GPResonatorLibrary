#[derive(Debug)]
pub struct ResonatorParams {
    pub w_0: f64,
    pub g: f64,
    pub r: f64,
}

impl ResonatorParams {
    #[inline]
    pub fn init(w_0: f64) -> Self {
        Self { w_0, g: 1.0, r: 0.5 }
    }

    #[inline]
    pub fn new(w_0: f64, g: f64, r: f64) -> Self {
        Self { w_0, g, r }
    }

    pub fn get_mag_at(&self, w: f64) -> f64 {
        let g = self.g;
        let r = self.r;
        let w_0 = self.w_0;

        g * r * w_0.sin() / (((2.0 * w).cos() - 2.0 * r * w.cos() * w_0.cos() + r.powi(2)).powi(2) + ((2.0 * w).sin() - 2.0 * r * w.sin() * w_0.cos()).powi(2)).sqrt()
    }
}