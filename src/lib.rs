


use rand;
use rand_distr::{Normal, Distribution};
use rand::distributions::WeightedIndex;

use serde::Deserialize;

/// Landmark struct used to hold both landmark and obervations
#[derive(Debug, Deserialize, Default, Clone)]
pub struct Landmark{
    pub x: f64,
    pub y: f64,
    pub id: u32
}

/// Vehicle controls; necessary for prediction step
#[derive(Debug)]
pub struct Controls {
    pub velocity: f64,
    pub yawrate: f64,
}

/// Struct containing a particle
#[derive(Debug, Default, Copy, Clone)]
pub struct Particle {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub phi: f64,
    pub weight: f64,
  }

/// Struct holding state and parameters of ParticleFilter
#[derive(Debug)]
pub struct ParticleFilter{
    n: u32,
    pub initialized: bool,
    particles: Vec<Particle>,
    pub best_particle: Particle,
    position_std: Vec<f64>, // uncertainty of (initial) GPS measurement
    lm_std: Vec<f64>,
    dt: f64,
    sensor_range: f64,
    epsilon: f64, // to avoid division by 0
    x_norm: Normal<f64>,
    y_norm: Normal<f64>,
    phi_norm: Normal<f64>,
}

impl ParticleFilter {
    pub fn new() -> ParticleFilter {
        ParticleFilter{
            n: 0,
            initialized: false,
            particles: vec![Particle {..Default::default()}],
            best_particle: Particle {..Default::default()},
            position_std : Vec::new(),
            lm_std: Vec::new(),
            dt: 0.,
            sensor_range: 0.,
            epsilon: 0.,
            x_norm: Normal::new(0.0, 0.).unwrap(),
            y_norm: Normal::new(0.0, 0.).unwrap(),
            phi_norm: Normal::new(0.0, 0.).unwrap(),
        }
    }
    pub fn init(&mut self, msmt: &Particle, gps_std: &Vec<f64>, lm_std: &Vec<f64>, dt: f64, sensor_range: f64, epsilon: f64, n: u32) {
        if self.initialized == true {
            return;
        }
        self.n = n;
        self.initialized = true;
        self.position_std = gps_std.clone();
        self.lm_std = lm_std.clone();
        self.dt = dt;
        self.sensor_range = sensor_range;
        self.epsilon = epsilon;
        self.x_norm = Normal::new(0.0, self.position_std[0]).unwrap();
        self.y_norm = Normal::new(0.0, self.position_std[1]).unwrap();
        self.phi_norm = Normal::new(0.0, self.position_std[2]).unwrap();
        self.particles.clear();
        for i in 1..1+self.n {
            // init particles 
            let particle = Particle {
                id: i,
                x: msmt.x + self.x_norm.sample(&mut rand::thread_rng()),
                y: msmt.y + self.y_norm.sample(&mut rand::thread_rng()),
                phi: msmt.phi + self.phi_norm.sample(&mut rand::thread_rng()),
                weight: 1.0,
                // ..Default::default()
            };
            self.particles.push(particle);
        }
        self.best_particle = Particle {..Default::default()};
    }
    /// Predict position using process model and delta t as well as controls
    pub fn predict(&mut self, controls: &Controls) {
        for mut p in &mut self.particles {
            if controls.yawrate.abs() < self.epsilon {
                p.x = p.x + controls.velocity * self.dt * p.phi.cos();
                p.y = p.y + controls.velocity * self.dt * p.phi.sin();
            } else {

                p.x += controls.velocity / controls.yawrate * ( (p.phi + controls.yawrate * self.dt).sin() - p.phi.sin() );
                p.y += controls.velocity / controls.yawrate * ( p.phi.cos() - (p.phi + controls.yawrate * self.dt).cos() );
                p.phi += controls.yawrate * self.dt;
            }
            p.x += self.x_norm.sample(&mut rand::thread_rng());
            p.y += self.y_norm.sample(&mut rand::thread_rng());
            p.phi += self.phi_norm.sample(&mut rand::thread_rng());
            p.weight = 1.;
        }
    }
    /// Associate current observations with closest landmark ID
    /// Currently not used; ToDo how to call method modifying struct within for loop of other method?
    // fn associate_obs_lm(&self, landmarks: &Vec<Landmark>, observations: &mut Vec<Landmark>) {
    //     for obs in observations {
    //         let mut min_dist = self.sensor_range * 2.0;
    //         let mut best_id: u32 = 0;

    //         for lm in landmarks {
    //             let current_dist = ((lm.x - obs.x).powi(2) + (lm.y - obs.y).powi(2)).sqrt();
    //             if current_dist < min_dist {
    //                 min_dist = current_dist;
    //                 best_id = lm.id;
    //             }
    //         obs.id = best_id;
    //         }
    //     }
    // }

    /// Updating particle weights based on multivariate distributions
    /// Taking predicted particles and observations to calculate weights
    /// Observations are in vehicle coordinates and are transformed into map coordinates
    pub fn update_weights(&mut self, observations: &Vec<Landmark>, landmarks: &Vec<Landmark>) {
        for p in &mut self.particles {
            // get all landmarks in sensor range
            let mut visible_lm  = Vec::new();
            let mut map_observations = Vec::new();

            for lm in landmarks {
                let dist = ( (lm.x - p.x).powi(2) + (lm.y - p.y).powi(2) ).sqrt();
                if dist <= self.sensor_range {
                    visible_lm.push(lm.clone());
                }
            }
            // transform observations into map coordinates
            for obs in observations {
                let m_obs = Landmark {
                    x: obs.x * p.phi.cos() - obs.y * p.phi.sin() + p.x,
                    y: obs.x * p.phi.sin() + obs.y * p.phi.cos() + p.y,
                    id: obs.id
                };
                map_observations.push(m_obs);
            }
            // associate lm to obs
            // self.associate_obs_lm(&visible_lm, &mut map_observations);
            for m_obs in &mut map_observations {
                let mut min_dist = self.sensor_range * 2.0;
                let mut best_id: u32 = 0;
    
                for lm in &visible_lm {
                    let current_dist = ((lm.x - m_obs.x).powi(2) + (lm.y - m_obs.y).powi(2)).sqrt();
                    if current_dist < min_dist {
                        min_dist = current_dist.clone();
                        best_id = lm.id.clone();
                    }
                }
                m_obs.id = best_id.clone();
            }
            // Calculate weight based on overlap of prob dist of obervation and closest landmark
            p.weight = 1.; // init with 1 for conv below
            for obs in &mut map_observations {
                if let Some(n_lm) = visible_lm.iter().find(|x| x.id == obs.id) {
                    let weight: f64 = (1.0 / (2.0 * std::f64::consts::PI * self.lm_std[0] * self.lm_std[1]) ) *
                        (-1.0*((obs.x - n_lm.x).powi(2) / (2.0 * self.lm_std[0]).powi(2) +
                        (obs.y - n_lm.y).powi(2) / (2.0 * self.lm_std[1]).powi(2) )).exp();
                    if weight < self.epsilon {
                        p.weight *= self.epsilon;
                    } else {
                        p.weight *= weight;
                    }
                } else {
                    println!("missing lm: {:?}", &obs);
                    p.weight = self.epsilon;
                }
            }
        }
    }
    
    /// resample particles with replacement proportional to weight
    pub fn resample(&mut self) {
        let mut new_particles = vec![];
        let weights: Vec<f64> = self.particles.iter().map(|x|  x.weight).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = rand::thread_rng();
        for i in 1..self.n+1 {
            let mut new_particle = self.particles[dist.sample(&mut rng)];
            new_particle.id = i;
            new_particles.push(new_particle);
        }
        self.particles = new_particles;
    }

    /// get particle with largest weight, most likely vehicle position
    pub fn get_best_particle(&mut self) {
        let current_weights: Vec<f64> = self.particles.iter().map(|x| x.weight).collect();
        // Can't get max of f64 in Rust, as floats are not ordered...
        // let max_weight = current_weights.iter().fold(None, |m, &x| m.map_or(Some(x), |mv| Some(if x > mv {x} else {mv})) ).unwrap();
        // This would be a better alternative for f64 - doesn't work for f64 as fold() not implemented
        // let max_weight = current_weights.iter().cloned().fold(0./0., std::f64::MAX);

        let mut max_weight: f64 = -1.;
        for w in current_weights {
            if w > max_weight {
                max_weight = w;
            }
        }

        let bp = self.particles.iter().find(|x| x.weight == max_weight).unwrap().clone();
        self.best_particle = bp;
    }

    /// returns error in x,y,phi estimate for best particle
    pub fn best_error(&self, ground_truth: &Particle) -> Vec<f64> {
        let x = (self.best_particle.x - ground_truth.x).abs();
        let y = (self.best_particle.y - ground_truth.y).abs();
        let phi = (self.best_particle.phi - ground_truth.phi).abs();
        vec![x, y, phi]
    }
}


