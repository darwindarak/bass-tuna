/// Identifies the fundamental frequency of a signal using the YIN algorithm.
///
/// # Arguments
/// * `input` - A slice of `f32` representing the audio signal samples.
/// * `sample_rate` - The sampling rate of the audio signal in Hz.
/// * `min_frequency` - The minimum frequency to detect (e.g., 20.0 Hz for low pitch).
/// * `max_frequency` - The maximum frequency to detect (e.g., 500.0 Hz for high pitch).
///
/// # Returns
/// * `Option<f32>` - The detected frequency in Hz, or `None` if no valid pitch is found.
///
pub fn identify_frequency(
    input: &[f32],
    sample_rate: f32,
    min_frequency: f32,
    max_frequency: f32,
) -> Option<f32> {
    // Check if "volume" exceeds threshold
    let mut volume = 0.0;
    for amplitude in input {
        volume += amplitude * amplitude;
    }
    let volume_threshold = 0.08;
    if volume < volume_threshold * volume_threshold {
        return None;
    }

    // Calculate the difference function d(t)
    let min_period = (sample_rate / max_frequency).floor() as usize;
    let max_period = (sample_rate / min_frequency).floor() as usize;

    let mut differences: Vec<f32> = vec![0.0; (max_period - min_period) + 1];
    for (i, period) in (min_period..=max_period).enumerate() {
        for j in 0..(input.len() - period) {
            let diff = input[j] - input[j + period];
            differences[i] += diff * diff;
        }
    }

    // Compute the cumulative mean normalized difference function
    let mut normed_differences = vec![0.0; differences.len()];
    let mut cumulative_sum = 0.0;

    for i in 1..differences.len() {
        cumulative_sum += differences[i];
        normed_differences[i] = differences[i] / (cumulative_sum / i as f32);
    }

    // Find a the first miniumum that is lower than some threshold
    let threshold = 0.1; // Typical YIN threshold for valid minima
    for i in 1..differences.len() - 1 {
        if normed_differences[i] < threshold
            && normed_differences[i] < normed_differences[i - 1]
            && normed_differences[i] < normed_differences[i + 1]
        {
            // Parabolic interpolation for sub-sample precision
            let dp = normed_differences[i - 1];
            let d = normed_differences[i];
            let dn = normed_differences[i + 1];

            let interpolation = (dp - dn) / (2.0 * (2.0 * d - dp - dn));
            let period = (i + min_period) as f32 + interpolation;

            // Convert period to frequency
            return Some(sample_rate / period);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng; // For random number generation
    use std::f32::consts::PI;

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

    #[test]
    fn test_pitch_detection_yin() {
        let fs = 44100.0; // Sampling rate in Hz
        let duration = 1.0; // Signal duration in seconds
        let min_frequency = 30.0;
        let max_frequency = 150.0;

        // Generate a random fundamental frequency in the range 30-150 Hz
        let mut rng = rand::thread_rng();
        let fundamental_frequency = rng.gen_range(min_frequency..max_frequency);

        // Generate the test signal with harmonics
        let signal = generate_sine_with_harmonics(fundamental_frequency, duration, fs, 3);

        // Run the pitch detection algorithm
        let detected_frequency = identify_frequency(&signal, fs, min_frequency, max_frequency);

        // Verify that the detected frequency is approximately the fundamental frequency
        if let Some(f_est) = detected_frequency {
            println!(
                "Actual frequency: {:.2} Hz, Detected frequency: {:.2} Hz",
                fundamental_frequency, f_est
            );
            assert_approx_eq::assert_approx_eq!(fundamental_frequency, f_est, 0.5);
        } else {
            panic!("Pitch detection failed to identify a frequency.");
        }
    }
}
