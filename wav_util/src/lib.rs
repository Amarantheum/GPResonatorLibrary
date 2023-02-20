use std::error::Error;
use std::path::Path;
use hound::SampleFormat;
use statrs::statistics::*;

/// Read a 2 channel wav file into an array of vectors
/// 
/// Returns the audio channels as well as the sample rate of the audio
pub fn read_wave<P:AsRef<Path>>(path: P) -> Result<([Vec<f64>; 2], usize), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let samples;
    let spec = reader.spec();
    match spec.sample_format {
        SampleFormat::Float => {
            samples = reader.samples::<f32>()
                .map(|v| {
                    v.unwrap() as f64
                })
                .collect::<Vec<f64>>();
        },
        SampleFormat::Int => {
            let max = 2_i32.pow((spec.bits_per_sample - 1) as u32);
            samples = reader.samples::<i32>()
                .map(|v| {
                    (v.unwrap() as f64) / max as f64
                })
                .collect::<Vec<f64>>();
        }
    }
    let mut chan1 = Vec::with_capacity(samples.len() / 2);
    let mut chan2 = Vec::with_capacity(samples.len() / 2);
    for i in 0..samples.len() {
        if i % 2 == 0 {
            chan1.push(samples[i])
        } else {
            chan2.push(samples[i])
        }
    }
    Ok(
        ([chan1, chan2], spec.sample_rate as usize)
    )
}

/// Given a set of two vectors representing audio channels to a wav file
pub fn write_wave<P: AsRef<Path>>(mut audio: [Vec<f64>; 2], path: P, sample_rate: u32) -> Result<(), Box<dyn Error>> {
    assert!(audio[0].len() == audio[1].len());
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    scale_audio(&mut audio);
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in audio[0].iter().zip(audio[1].iter()) {
        writer.write_sample(*s.0 as f32)?;
        writer.write_sample(*s.1 as f32)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Normalizes the signal if the maximum amplitude is greater than 1.0 (uses the absolute value of the amplitude)
pub fn scale_audio(audio: &mut [Vec<f64>; 2]) {
    let max = [(&audio[0]).abs_max(), (&audio[1]).abs_max()].max();
    if max <= 1.0 {
        return;
    }
    for v in &mut audio[0] {
        *v /= max;
    }
    for v in &mut audio[1] {
        *v /= max;
    }
}

pub fn combine_channels(audio: &[Vec<f64>]) -> Vec<f64> {
    let div = audio.len();
    let mut out = Vec::with_capacity(audio[0].len());
    for a in audio {
        for (i, s) in a.iter().enumerate() {
            out[i] = *s;
        }
    }
    out
}

