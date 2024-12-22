use std::f32::consts::PI;

/// A collection of resonators tuned to specific frequencies.
pub struct Resonators {
    frequencies: Vec<f32>,           // Target frequencies for the resonators
    filters: Vec<BiquadFilter>,      // Filters tuned to the target frequencies
    energy_histories: Vec<Vec<f32>>, // Historical energy values for each filter
    energies: Vec<f32>,              // Current energy levels for each filter
    n_energy_history: usize,         // Number of historical energy samples to track
    current_energy_index: usize,     // Index for circular buffer management
}

impl Resonators {
    /// Creates a new Resonators instance with specified frequencies, sample rate, and quality factor.
    ///
    /// # Arguments
    /// * `frequencies` - A slice of target frequencies in Hz.
    /// * `sample_rate` - The sample rate of the audio in Hz.
    /// * `q` - The quality factor for the resonators.
    /// * `n_energy_history` - The size of the energy history buffer.
    pub fn new(frequencies: &[f32], sample_rate: i32, q: f32, n_energy_history: usize) -> Self {
        let filters: Vec<BiquadFilter> = frequencies
            .iter()
            .map(|&f| BiquadFilter::new(f, sample_rate, q))
            .collect();

        let energy_histories: Vec<Vec<f32>> = vec![vec![0.0; n_energy_history]; frequencies.len()];
        let energies = vec![0.0; frequencies.len()];

        Self {
            frequencies: frequencies.to_vec(),
            filters,
            energy_histories,
            energies,
            n_energy_history,
            current_energy_index: 0,
        }
    }

    /// Returns the frequency and energy of the current peak resonator.
    ///
    /// # Returns
    /// A tuple containing the frequency and energy of the peak.
    pub fn current_peak(&self) -> (f32, f32) {
        self.energies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ind, &energy)| (self.frequencies[ind], energy))
            .unwrap_or((0.0, 0.0))
    }

    /// Processes a new batch of audio samples and updates the resonator energies.
    ///
    /// # Arguments
    /// * `samples` - A slice of input audio samples.
    pub fn process_new_samples(&mut self, samples: &[f32]) {
        for &sample in samples {
            for (n_filter, filter) in self.filters.iter_mut().enumerate() {
                let y = filter.apply(sample);
                let energy = y * y;

                // Update running energy using the circular buffer
                self.energies[n_filter] +=
                    energy - self.energy_histories[n_filter][self.current_energy_index];
                self.energy_histories[n_filter][self.current_energy_index] = energy;
            }
            self.current_energy_index = (self.current_energy_index + 1) % self.n_energy_history;
        }
    }
}

/// A biquad filter implementation for resonator filtering.
struct BiquadFilter {
    b_0: f32, // Feedforward coefficient
    b_1: f32,
    b_2: f32,
    a_1: f32, // Feedback coefficient
    a_2: f32,

    y_1: f32, // Previous output sample (y[n-1])
    y_2: f32, // Output sample (y[n-2])
    x_1: f32, // Previous input sample (x[n-1])
    x_2: f32, // Input sample (x[n-2])
}

impl BiquadFilter {
    /// Applies the filter to a single input sample and returns the output sample.
    ///
    /// # Arguments
    /// * `x` - The input sample.
    ///
    /// # Returns
    /// The filtered output sample.
    fn apply(&mut self, x: f32) -> f32 {
        let y = self.b_0 * x + self.b_1 * self.x_1 + self.b_2 * self.x_2
            - self.a_1 * self.y_1
            - self.a_2 * self.y_2;

        // Update state variables
        self.y_2 = self.y_1;
        self.y_1 = y;
        self.x_2 = self.x_1;
        self.x_1 = x;

        y
    }

    /// Creates a new biquad filter tuned to a specific frequency.
    ///
    /// # Arguments
    /// * `f0` - The target frequency in Hz.
    /// * `sample_rate` - The sample rate of the audio in Hz.
    /// * `q` - The quality factor of the filter.
    ///
    /// # Returns
    /// A new `BiquadFilter` instance.
    pub fn new(f0: f32, sample_rate: i32, q: f32) -> Self {
        let k = f0 / sample_rate as f32;
        let omega_0 = 2.0 * PI * k;
        let alpha = omega_0.sin() / (2.0 * q);

        let a_0 = 1.0 + alpha;
        let a_1 = -2.0 * omega_0.cos() / a_0;
        let a_2 = (1.0 - alpha) / a_0;

        let b_0 = alpha / a_0;
        let b_1 = 0.0;
        let b_2 = -b_0;

        Self {
            b_0,
            b_1,
            b_2,
            a_1,
            a_2,
            y_1: 0.0,
            y_2: 0.0,
            x_1: 0.0,
            x_2: 0.0,
        }
    }
}

/// Identifies the musical note name, pitch, and cents offset for a given frequency.
///
/// # Arguments
/// * `f` - The frequency in Hz to identify.
///
/// # Returns
/// A tuple containing:
/// * The name of the note (e.g., "A", "C", "G#").
/// * The pitch of the note in Hz.
/// * The offset in cents relative to the given frequency.
pub fn identify_note_name(f: f32) -> (String, f32, f32) {
    // Calculate the MIDI note number (A440 corresponds to MIDI note 49)
    let n = (12.0 * (f / 440.0).log2()).round() as i32 + 49;

    // Determine the note name using the MIDI note number modulo 12
    let name = match n % 12 {
        0 => "Ab",
        1 => "A",
        2 => "Bb",
        3 => "B",
        4 => "C",
        5 => "Db",
        6 => "D",
        7 => "Eb",
        8 => "E",
        9 => "F",
        10 => "Gb",
        11 => "G",
        _ => panic!("Unexpected note number: {}", n),
    }
    .to_string();

    // Calculate the exact pitch of the note
    let pitch = 440.0 * 2.0f32.powf((n as f32 - 49.0) / 12.0);

    // Calculate the cents offset from the given frequency
    let cents = 1200.0 * (f / pitch).log2();

    (name, pitch, cents)
}

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
    interpolate: bool,
) -> Option<f32> {
    // Check if "volume" exceeds threshold
    let mut volume = 0.0;
    for amplitude in input {
        volume += amplitude * amplitude;
    }
    let volume_threshold = 0.01;
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
    let threshold = 0.25;
    for i in 1..differences.len() - 1 {
        if normed_differences[i] < threshold
            && normed_differences[i] < normed_differences[i - 1]
            && normed_differences[i] < normed_differences[i + 1]
        {
            let period = if interpolate {
                // Parabolic interpolation for sub-sample precision
                let dp = normed_differences[i - 1];
                let d = normed_differences[i];
                let dn = normed_differences[i + 1];

                let interpolation = (dp - dn) / (2.0 * (2.0 * d - dp - dn));
                (i + min_period) as f32 + interpolation
            } else {
                (i + min_period) as f32
            };

            // Convert period to frequency
            return Some(sample_rate / period);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use rand::Rng;
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
            let amplitude = rng.gen_range(0.1..0.3); // Random amplitude for harmonics
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
        let detected_frequency =
            identify_frequency(&signal, fs, min_frequency, max_frequency, false);

        // Verify that the detected frequency is approximately the fundamental frequency
        if let Some(f_est) = detected_frequency {
            println!(
                "Actual frequency: {:.2} Hz, Detected frequency: {:.2} Hz",
                fundamental_frequency, f_est
            );
            assert_approx_eq!(fundamental_frequency, f_est, 0.5);
        } else {
            panic!("Pitch detection failed to identify a frequency.");
        }
    }

    #[test]
    fn test_resonator_pitch_detection_yin() {
        let fs = 44100.0; // Sampling rate in Hz
        let duration = 1.0; // Signal duration in seconds
        let min_frequency = 30.0;
        let max_frequency = 150.0;

        // Generate a random fundamental frequency in the range 30-150 Hz
        let mut rng = rand::thread_rng();
        let fundamental_frequency = rng.gen_range(min_frequency..max_frequency);

        // Generate the test signal with harmonics
        let signal = generate_sine_with_harmonics(fundamental_frequency, duration, fs, 3);

        let mut resonators = Resonators::new(
            &[30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
            fs as i32,
            10.0,
            fs as usize / 20,
        );
        resonators.process_new_samples(&signal);
        let (freq, _) = resonators.current_peak();
        let detected_frequency = identify_frequency(&signal, fs, freq - 10.0, freq + 10.0, true);

        // Verify that the detected frequency is approximately the fundamental frequency
        if let Some(f_est) = detected_frequency {
            println!(
                "Actual frequency: {:.2} Hz, Detected frequency: {:.2} Hz",
                fundamental_frequency, f_est
            );
            assert_approx_eq!(fundamental_frequency, f_est, 0.5);
        } else {
            panic!("Pitch detection failed to identify a frequency.");
        }
    }

    #[test]
    fn test_identify_note_name_exact() {
        let (name, pitch, cents) = identify_note_name(440.0);
        assert_eq!(name, "A");
        assert!((pitch - 440.0).abs() < 1e-6);
        assert!((cents - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_identify_note_name_sharp() {
        let (name, pitch, cents) = identify_note_name(466.1638);
        assert_eq!(name, "Bb");
        assert_approx_eq!(pitch, 466.1638, 1e-6);
        assert_approx_eq!(cents, 0.00, 1e-2);
    }

    #[test]
    fn test_identify_note_name_flat() {
        let (name, pitch, cents) = identify_note_name(415.3047);
        assert_eq!(name, "Ab");
        assert_approx_eq!(pitch, 415.3047, 1e-6);
        assert_approx_eq!(cents, 0.00, 1e-2);
    }

    #[test]
    fn test_identify_note_name_cents_offset() {
        let (name, pitch, cents) = identify_note_name(445.0);
        assert_eq!(name, "A");
        assert_approx_eq!(pitch, 440.0, 1e-6);
        assert_approx_eq!(cents, 19.56, 1e-2);
    }
}
