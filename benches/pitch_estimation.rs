use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng; // For random number generation
use std::f32::consts::PI;

use lib::tuner::{identify_frequency, Resonators};

/// Generate a synthetic sine wave signal with harmonics.
/// - `f` is the fundamental frequency.
/// - `duration` is the signal length in seconds.
/// - `fs` is the sampling rate in Hz.
/// - `harmonics` is the number of additional harmonics to include.
fn generate_sine_with_harmonics(f: f32, duration: f32, fs: f32, harmonics: usize) -> Vec<f32> {
    let num_samples = (duration * fs) as usize;
    let mut signal = vec![0.0; num_samples];
    let dt = 1.0 / fs;

    // Add the fundamental frequency
    for i in 0..num_samples {
        signal[i] += (2.0 * PI * f * i as f32 * dt).sin();
    }

    // Add harmonics
    let mut rng = rand::thread_rng();
    for _ in 0..harmonics {
        let amplitude = rng.gen_range(0.2..1.0); // Random amplitude for harmonics
        let harmonic_multiplier = rng.gen_range(2..6); // Random integer multiple of the fundamental
        for i in 0..num_samples {
            signal[i] +=
                amplitude * (2.0 * PI * f * harmonic_multiplier as f32 * i as f32 * dt).sin();
        }
    }

    signal
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let fs = 44100.0; // Sampling rate in Hz
    let duration = 1.0; // Signal duration in seconds
    let min_frequency = 30.0;
    let max_frequency = 150.0;

    // Generate a random fundamental frequency in the range 30-150 Hz
    let mut rng = rand::thread_rng();
    let fundamental_frequency = rng.gen_range(min_frequency..max_frequency);

    // Generate the test signal with harmonics
    let signal = generate_sine_with_harmonics(fundamental_frequency, duration, fs, 3);

    c.bench_function("identify_frequency", |b| {
        b.iter(|| {
            let mut resonators = Resonators::new(
                &[30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
                fs as i32,
                10.0,
                fs as usize / 20,
            );
            resonators.process_new_samples(&signal);
            let (freq, _) = resonators.current_peak();
            identify_frequency(&signal, fs, freq - 10.0, freq + 10.0, true);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
