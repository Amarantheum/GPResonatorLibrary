use rand::Rng;

/// Expectation maximization algorithm for fitting Gaussian curves to a function. Uses a Gaussian Mixture Model to fit the data.
pub struct GaussianEstimator {
    x: Vec<f64>,
    y: Vec<f64>,
    y_sum: f64,
    gaussians: Vec<Gaussian>,
}

impl GaussianEstimator {
    pub fn new(x: Vec<f64>, y: Vec<f64>, num_cluster: usize) -> Self {
        assert_eq!(x.len(), y.len());
        let y_sum = y.iter().sum::<f64>();
        let mut gaussians = Vec::with_capacity(num_cluster);
        let mut rng = rand::thread_rng();
        for _ in 0..num_cluster {
            let mean = rng.gen_range(0.0..std::f64::consts::PI);
            let variance = 0.1;
            let weight = 1.0 / num_cluster as f64;
            gaussians.push(Gaussian::new(mean, variance, weight));
        }
        Self {
            x,
            y,
            y_sum,
            gaussians,
        }
    }

    pub fn new_with_initial(x: Vec<f64>, y: Vec<f64>, gaussians: Vec<Gaussian>) -> Self {
        assert_eq!(x.len(), y.len());
        let y_sum = y.iter().sum::<f64>();
        Self {
            x,
            y,
            y_sum,
            gaussians,
        }
    }

    fn compute_responsibilities(&self, gaussians: &[Gaussian]) -> Vec<Vec<f64>> {
        let mut responsibilities = Vec::with_capacity(gaussians.len());
        let log_pdf_table = gaussians.iter().map(|g| self.x.iter().map(|x| g.log_pdf(*x)).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();
        let mut max_log_x_table = Vec::with_capacity(self.x.len());
        for i in 0..self.x.len() {
            let max_log_x = log_pdf_table.iter().map(|v| v[i]).fold(f64::NEG_INFINITY, |acc, v| acc.max(v));
            max_log_x_table.push(max_log_x);
        }
        for g in 0..gaussians.len() {
            let mut responsibility = Vec::with_capacity(self.x.len());
            for i in 0..self.x.len() {
                let log_numerator = log_pdf_table[g][i];
                let max_x = max_log_x_table[i];
                let log_responsibility = log_numerator - max_x - (log_pdf_table.iter().map(|v| v[i] - max_x).fold(0.0, |acc, v| acc + v.exp())).ln();
                responsibility.push(log_responsibility.exp() * self.y[i]);
            }
            responsibilities.push(responsibility);
        }

        responsibilities
    }

    pub fn compute_l_infty(g1: &[Gaussian], g2: &[Gaussian]) -> (f64, f64) {
        let mean_max = g1.iter().zip(g2.iter()).map(|(g1, g2)| (g1.mean - g2.mean).abs()).fold(0.0, |acc: f64, v| acc.max(v));
        let variance_max = g1.iter().zip(g2.iter()).map(|(g1, g2)| (g1.variance - g2.variance).abs()).fold(0.0, |acc: f64, v| acc.max(v));
        (mean_max, variance_max)
    }

    pub fn estimate(&self, iterations: usize) -> Vec<Gaussian> {
        let mut rng = rand::thread_rng();
        let mut gaussians = self.gaussians.clone();

        for iteration in 0..iterations {
            let mut cloned = gaussians.clone();
            cloned.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

            let responsibilities = self.compute_responsibilities(&gaussians);
            //println!("{:?}", responsibilities);
            let mut new_gaussians = Vec::with_capacity(self.gaussians.len());
            for i in 0..self.gaussians.len() {
                let mut new_mean = 0.0;
                let mut new_variance = 0.0;
                let mut new_weight = 0.0;
                for j in 0..self.x.len() {
                    new_mean += responsibilities[i][j] * self.x[j];
                    new_variance += responsibilities[i][j] * (self.x[j] - gaussians[i].mean).powi(2);
                    new_weight += responsibilities[i][j];
                }
                new_mean /= new_weight;
                new_variance /= new_weight;
                new_weight /= self.y_sum;
                new_gaussians.push(Gaussian::new(new_mean, new_variance, new_weight));
            }
            let (mean_max, variance_max) = Self::compute_l_infty(&cloned, &new_gaussians);
            println!("Iteration: {}, Mean Max: {}, Variance Max: {}", iteration, mean_max, variance_max);
            if mean_max < 1e-6 && variance_max < 1e-6 {
                break;
            }

            gaussians = new_gaussians;
        }

        gaussians
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    mean: f64,
    variance: f64,
    weight: f64,
}

impl Gaussian {
    pub fn new(mean: f64, variance: f64, weight: f64) -> Self {
        Self { mean, variance, weight }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let a = 1.0 / (2.0 * std::f64::consts::PI * self.variance).sqrt();
        let b = (x - self.mean).powi(2) / (2.0 * self.variance);
        self.weight * a * (-b).exp()
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        let a = 1.0 / (2.0 * std::f64::consts::PI * self.variance).sqrt();
        let b = (x - self.mean).powi(2) / (2.0 * self.variance);
        self.weight.ln() + a.ln() - b
    }
}

#[cfg(test)]
mod em_tests {
    use super::*;
    use std::f64::consts::PI;
    use plotters::prelude::*;
    use crate::resonance_params::ResonatorParams;

    fn plot_x_y(x: &Vec<f64>, y: &Vec<f64>, name: &str, graph_title: &str) {
        let y_max = y.iter().fold(0.0, |acc: f64, &v| acc.max(v));

        let root = BitMapBackend::new(name, (1920, 1080)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption(graph_title, ("sans-serif", 30).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..PI, 0.0..y_max)
            .unwrap();
        chart.configure_mesh().draw().unwrap();
        chart.draw_series(LineSeries::new(x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)), &RED)).unwrap();
    }

    #[test]
    fn test_gaussian() {
        let gaussian = Gaussian::new(0.0, 1.0, 1.0);
        assert!((gaussian.pdf(0.0) - 0.3989422804014327).abs() < 1e-6);

        let gaussian = Gaussian::new(0.0, 1.0, 3.0);
        assert!((gaussian.pdf(0.0) - 0.3989422804014327 * 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_estimator_basic() {
        const X_SIZE: usize = 1024;
        let mut gaussians = Vec::new();
        gaussians.push(Gaussian::new(0.5, 0.01, 1.0));
        gaussians.push(Gaussian::new(1.5, 0.01, 1.0));
        gaussians.push(Gaussian::new(2.0, 0.01, 1.0));
        
        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum);
        }
        plot_x_y(&x, &y, "gaussian_basic.png", "Gaussian Basic");

        let estimator = GaussianEstimator::new(x, y, 10);
        let estimated_gaussians = estimator.estimate(100);
        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &estimated_gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum);
        }
        plot_x_y(&x, &y, "gaussian_estimated_basic.png", "Gaussian Estimated Basic");
    }

    #[test]
    fn test_gaussian_estimator() {
        let mut rng = rand::thread_rng();
        const TEST_SIZE: usize = 500;
        const X_SIZE: usize = 1024;
        let mut gaussians = Vec::with_capacity(TEST_SIZE);
        for _ in 0..TEST_SIZE {
            let mean = rng.gen_range(0.0..PI);
            let variance = rng.gen_range(0.00001..0.0001);
            let weight = rng.gen_range(0.1..1.0);
            gaussians.push(Gaussian::new(mean, variance, weight));
        }
        gaussians.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        println!("{:?}", gaussians);

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum + rng.gen_range(-5.0..5.0));
        }
        plot_x_y(&x, &y, "gaussian.png", "Gaussian");

        let len = gaussians.len();
        let est_gaussians = gaussians.iter().map(|g| Gaussian::new(g.mean, 0.00001, 1.0 / len as f64)).collect::<Vec<_>>();
        let estimator = GaussianEstimator::new_with_initial(x, y, est_gaussians);
        let mut estimated_gaussians = estimator.estimate(100);
        estimated_gaussians.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        println!("{:?}", estimated_gaussians);

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &estimated_gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum);
        }
        plot_x_y(&x, &y, "gaussian_estimated.png", "Gaussian Estimated");
    }

    #[test]
    fn test_gaussian_estimator_modes_basic() {
        const X_SIZE: usize = 1024;

        let mut resonance_params = Vec::new();
        resonance_params.push(ResonatorParams::new(0.5, 1.0, 0.999));
        resonance_params.push(ResonatorParams::new(1.0, 1.0, 0.999));
        resonance_params.push(ResonatorParams::new(1.5, 1.0, 0.999));
        resonance_params.push(ResonatorParams::new(2.0, 1.0, 0.999));
        resonance_params.push(ResonatorParams::new(2.5, 1.0, 0.999));

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for params in &resonance_params {
                sum += params.get_mag_at(x_);
            }
            x.push(x_);
            y.push(sum);
        }
        let min_y = y.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
        for y in &mut y {
            *y = (*y - min_y).max(0.0);
        }

        plot_x_y(&x, &y, "gaussian_modes_basic.png", "Original Frequency Response");

        let len = resonance_params.len();
        let est_gaussians = resonance_params.iter().map(|p| Gaussian::new(p.w_0, 0.0001, 1.0 / len as f64)).collect::<Vec<_>>();

        let estimator = GaussianEstimator::new_with_initial(x, y, est_gaussians);
        let estimated_gaussians = estimator.estimate(100);

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);

        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &estimated_gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum);
        }

        plot_x_y(&x, &y, "gaussian_estimated_modes_basic.png", "GMM Estimated Frequency Response");
    }

    #[test]
    fn test_gaussian_estimator_modes() {
        const TEST_SIZE: usize = 100;
        const X_SIZE: usize = 2048;

        let mut rng = rand::thread_rng();
        let mut resonance_params = Vec::with_capacity(TEST_SIZE);
        for _ in 0..TEST_SIZE {
            let w_0 = rng.gen_range(0.0..PI);
            let g = rng.gen_range(0.1..1.0);
            let r = rng.gen_range(0.99..0.999);
            resonance_params.push(ResonatorParams::new(w_0, g, r));
        }

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);
        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for params in &resonance_params {
                sum += params.get_mag_at(x_);
            }
            x.push(x_);
            y.push(sum); //+ rng.gen_range(-5.0..5.0));
        }
        let min_y = y.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
        for y in &mut y {
            *y = (*y - min_y).max(0.0);
        }

        plot_x_y(&x, &y, "gaussian_modes.png", "Original Frequency Response");

        let len = resonance_params.len();
        let est_gaussians = resonance_params.iter().map(|p| Gaussian::new(p.w_0, 0.00001, 1.0 / len as f64)).collect::<Vec<_>>();
        let estimator = GaussianEstimator::new_with_initial(x, y, est_gaussians);
        let estimated_gaussians = estimator.estimate(100);

        let mut x = Vec::with_capacity(X_SIZE);
        let mut y = Vec::with_capacity(X_SIZE);

        for i in 0..X_SIZE {
            let mut sum = 0.0;
            let x_ = i as f64 / X_SIZE as f64 * PI;
            for gaussian in &estimated_gaussians {
                sum += gaussian.pdf(x_);
            }
            x.push(x_);
            y.push(sum);
        }

        plot_x_y(&x, &y, "gaussian_estimated_modes.png", "GMM Estimated Frequency Response");
    }
}