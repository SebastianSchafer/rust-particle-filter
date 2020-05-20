
use std::{
        fs,
        process,
        time,
    };

use csv;

mod tools;
use tools::{read_map, 
            read_observations, 
            read_controls,
            read_ground_truth};

mod lib;
pub use lib::{
        Landmark, 
        Controls, 
        Particle,
        ParticleFilter};


fn main() {
    // manually set parameters - SI units
    let number_of_particles = 42;
    let sensor_range: f64 = 50.0;
    let dt: f64 = 0.1;

    let gps_std: Vec<f64> = vec![0.3, 0.3, 0.01]; // error of initial poitioning
    let landmark_std: Vec<f64> = vec![0.3, 0.3]; // msmt/process error

    let epsilon: f64 = 1e-5;

    // read in all data first:
    let map_file = "data/map_data.txt";
    let obs_folder = "data/observation/";
    let controls_file = "data/control_data.txt";
    let gt_file = "data/ground_truth_data.txt";

    let mut landmarks: Vec<Landmark> = Vec::new();
    let mut all_obs: Vec<Vec<Landmark>> = Vec::new();
    let mut controls: Vec<Controls> = Vec::new();
    let mut ground_truths: Vec<Particle> = Vec::new();


    if let Err(e) = read_map(map_file, &mut landmarks) {
        eprintln!("Application Error: {}", e);
        process::exit(1);
    }
    if let Err(e) = read_observations(obs_folder, &mut all_obs) {
        eprintln!("Application Error: {}", e);
        process::exit(1);
    }
    if let Err(e) = read_controls(controls_file, &mut controls) {
        eprintln!("Application Error: {}", e);
        process::exit(1);
    }
    if let Err(e) = read_ground_truth(gt_file, &mut ground_truths) {
        eprintln!("Application Error: {}", e);
        process::exit(1);
    }
    println!("Finished reading in data.");
    let t_start = time::Instant::now();
    // logging data for debug
    let mut log: Vec<Vec<f64>> = Vec::new();

    let mut pf = ParticleFilter::new();

    for (i, observations) in all_obs.iter().enumerate() {
        if pf.initialized == false {
            pf.init(&ground_truths[i], &gps_std, &landmark_std, dt, sensor_range, epsilon, number_of_particles);
        } else {
            // predict position based on current state, dt, and controls
            pf.predict(&controls[i-1]);

        }

        // update particle weights based on new msmt
        pf.update_weights(observations, &landmarks);

        pf.resample();

        // Best estimate of vehicle position
        pf.get_best_particle();
        // let distance = pf.best_error(&ground_truths[i]);

        // for log output
        log.push(vec![pf.best_particle.x, 
                    pf.best_particle.y,
                    pf.best_particle.phi,
                    ground_truths[i].x,
                    ground_truths[i].y,
                    ground_truths[i].phi]);
    }
    let duration = t_start.elapsed();
    println!("Main loop took {:#?}s", duration);

    // print out log
    let f = fs::OpenOptions::new().write(true).append(false).open("log.csv").expect("Issue creating log file.");
    let mut wtr = csv::WriterBuilder::new().from_writer(f);
    wtr.write_record(&["pred_x", "pred_y", "pred_phi", "true_X", "true_y", "true_phi"]).ok();

    for line in log {
        wtr.serialize(line).ok();
    }
    wtr.flush().ok();
    println!("Finished writing log.")
    
}




