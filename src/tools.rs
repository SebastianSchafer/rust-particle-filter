
use std::{io, fs, error::Error};
use serde::Deserialize;

use crate::{Landmark, Controls, Particle};


pub fn read_map(fname: &str, landmarks: &mut Vec<Landmark>) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(fname)?;
    for line in rdr.deserialize() {
        let lm: Landmark = line?;
        landmarks.push(lm);                    
    }
    Ok(())
}

pub fn read_controls(fname: &str, controls: &mut Vec<Controls>) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(fname)?;
    for line in text.lines() {
        let control: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        controls.push(Controls{velocity:control[0], yawrate:control[1]});
    }

    Ok(())
}


fn read_next_obs_file(fname: &str, observations: &mut Vec<Landmark>) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(fname)?;

    for line in text.lines() {
        let obs: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        observations.push(Landmark{x:obs[0], y:obs[1], id:0});
    }
    Ok(())
}

pub fn read_observations(path: &str, all_obs: &mut Vec<Vec<Landmark>>) -> Result<(), Box<dyn Error>> {
    let mut fnames = fs::read_dir(path)?
                                    .map(|x| x.map(|y|y.path()))
                                    .collect::<Result<Vec<_>, io::Error>>()?;
    fnames.sort();
    for fname in fnames {
        let mut observations: Vec<Landmark> = Vec::new();
        read_next_obs_file(fname.to_str().unwrap(), &mut observations)?;
        all_obs.push(observations);
    }                                
    println!("Number of observation steps: {:?}", all_obs.len());


    Ok(())
}


pub fn read_ground_truth(fname: &str, positions: &mut Vec<Particle>) -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string(fname)?;
    for line in text.lines() {
        let position: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        positions.push(Particle{x:position[0], y:position[1], phi:position[2], ..Default::default()});
    }

    Ok(())
}