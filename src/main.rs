use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{
    cursor::MoveToColumn,
    execute,
    style::{PrintStyledContent, Stylize},
    terminal::{Clear, ClearType},
};
use std::io::stdout;
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
    thread,
};

/// A fixed-size circular buffer
/// - `buffer`: The underlying storage for the circular buffer.
/// - `max_size`: The maximum capacity of the buffer.
/// - `write_index`: The position in the buffer where the next chunk will be written.
struct CircularBuffer {
    buffer: Vec<f32>,
    max_size: usize,
    write_index: usize,
}

impl CircularBuffer {
    /// Creates a new circular buffer with a specified maximum size.
    ///
    /// # Arguments
    /// * `max_size` - The capacity of the buffer (number of elements it can hold).
    ///
    /// # Returns
    /// A `CircularBuffer` instance initialized with zeros.
    fn new(max_size: usize) -> Self {
        Self {
            buffer: vec![0.0; max_size],
            max_size,
            write_index: 0,
        }
    }
    /// Adds a chunk of data to the circular buffer.
    ///
    /// If the chunk is larger than the buffer size, only the last `max_size` elements are stored.
    /// If the chunk wraps around the end of the buffer, it is split and written in two parts.
    ///
    /// # Arguments
    /// * `chunk` - A slice of `f32` values to add to the buffer.
    fn add_chunk(&mut self, chunk: &[f32]) {
        if chunk.len() < self.max_size {
            // Case 1: Chunk can fit into the buffer without overflow
            let remaining = self.max_size - self.write_index;

            if remaining >= chunk.len() {
                self.buffer[self.write_index..(self.write_index + chunk.len())]
                    .copy_from_slice(&chunk);
                self.write_index += chunk.len();
            } else {
                // The chunk wraps around the buffer
                // 1. Copy the first part to the end of the buffer
                self.buffer[self.write_index..].copy_from_slice(&chunk[..remaining]);

                // 2. Copy the remainder to the beginning of the buffer
                let wrapped_length = chunk.len() - remaining;
                self.buffer[..wrapped_length].copy_from_slice(&chunk[remaining..]);

                // Update the write index to the new position
                self.write_index = wrapped_length;
            }
        } else {
            // Case 2: Chunk is larger than the buffer size
            // Only the last `max_size` elements are retained
            let start = chunk.len() - self.max_size;
            self.write_index = 0;
            self.buffer.clone_from_slice(&chunk[start..]);
        }
    }

    /// Copies the current state of the circular buffer into an output slice.
    ///
    /// The data is copied in order, starting from the oldest data to the newest.
    ///
    /// # Arguments
    /// * `output` - A mutable slice where the buffer contents will be copied.
    ///
    /// # Panics
    /// Panics if `output` is not the same length as the circular buffer (`max_size`).
    fn copy_to_buffer(&self, output: &mut [f32]) {
        assert_eq!(output.len(), self.max_size, "Output slice size mismatch");

        let start_len = self.max_size - self.write_index;
        output[..start_len].copy_from_slice(&self.buffer[self.write_index..]);
        output[start_len..].copy_from_slice(&self.buffer[..self.write_index]);
    }

    /// Returns a copy of the current state of the buffer as a vector.
    ///
    /// The data is ordered, starting from the oldest to the newest values.
    ///
    /// # Returns
    /// A `Vec<f32>` containing the buffer contents.
    fn get_buffer(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.max_size);
        result.extend_from_slice(&self.buffer[self.write_index..]);
        result.extend_from_slice(&self.buffer[..self.write_index]);
        result
    }
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
/// # Example Usage
/// ```
/// let input = vec![0.1, 0.2, 0.1, -0.1, -0.2, -0.1]; // Example signal
/// let sample_rate = 44100.0;
/// let frequency = identify_frequency(&input, sample_rate, 20.0, 500.0);
/// if let Some(freq) = frequency {
///     println!("Detected Frequency: {:.2} Hz", freq);
/// } else {
///     println!("No frequency detected.");
/// }
/// ```
fn identify_frequency(
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

    let mut differences: Vec<f32> = vec![0.0; (max_period - min_period) as usize + 1];
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
    return None;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let devices: Vec<_> = host.input_devices()?.collect();
    let device_names: Vec<String> = devices
        .iter()
        .map(|d| d.name().unwrap_or("Unknown".to_string()))
        .collect();

    println!("Available input devices:");
    for (i, name) in device_names.iter().enumerate() {
        println!("{}: {}", i + 1, name);
    }

    print!("Select an input device by number: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let index: usize = input.trim().parse()?;

    let device = devices
        .get(index - 1)
        .expect("Invalid input device selection");

    let config = device.default_input_config()?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let max_window = sample_rate / 10; // 2 seconds
    let options = vec![41.2, 55.0, 73.4, 98.0]; // E1, A1, D2, G2 frequencies

    let circular_buffer = Arc::new(Mutex::new(CircularBuffer::new(max_window as usize))); // Share buffer between threads
    let process_buffer = Arc::clone(&circular_buffer);
    // Thread to process pitch detection
    thread::spawn(move || {
        let mut work_buffer = vec![0.0; max_window as usize];
        loop {
            // Safely access and retrieve a linearized copy of the buffer
            {
                let buf = process_buffer.lock().unwrap();
                buf.copy_to_buffer(&mut work_buffer);
            }
            let pitches = vec!["E", "A", "D", "G"];

            //let (pitch, energy, closest_choice) =
            //    identify_pitch(&work_buffer, &options, sample_rate as f32, 5, 5.0);
            let output = if let Some(freq) =
                identify_frequency(&work_buffer, sample_rate as f32, 30.0, 150.0)
            {
                let mut closest_choice = 0;
                let mut smallest_difference = 10000f32;
                for (i, f_ref) in options.iter().enumerate() {
                    let delta = (f_ref - freq).abs();
                    if delta < smallest_difference {
                        closest_choice = i;
                        smallest_difference = delta;
                    }
                }
                let target_pitch = options[closest_choice];
                let cents = 1200.0 * (freq / target_pitch).log2();
                let within_range = cents < 2.0;
                if within_range {
                    // Print in green if within range
                    format!(
                        "Detected pitch {} ({:.2}): {:.2} Hz ({} cents)",
                        pitches[closest_choice], target_pitch, freq, cents
                    )
                    .green()
                } else {
                    // Print in yellow if out of range
                    format!(
                        "Detected pitch {} ({:.2}): {:.2} Hz ({} cents)",
                        pitches[closest_choice], target_pitch, freq, cents
                    )
                    .yellow()
                }
            } else {
                format!("None found").red()
            };
            let mut stdout = stdout();
            execute!(stdout, MoveToColumn(0), Clear(ClearType::CurrentLine)).unwrap();
            execute!(stdout, PrintStyledContent(output)).unwrap();
            stdout.flush().unwrap();

            thread::sleep(std::time::Duration::from_millis(100)); // Control processing frequency
        }
    });

    let input_buffer = Arc::clone(&circular_buffer);
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            let mono_chunk: Vec<f32> = data
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect();
            let mut buf = input_buffer.lock().unwrap();
            buf.add_chunk(&mono_chunk);
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;

    stream.play()?;

    println!("Recording and processing audio. Press Ctrl+C to stop.");
    loop {
        std::thread::park(); // Keep the main thread alive
    }
}

#[cfg(test)]
mod tests {
    use super::{identify_frequency, CircularBuffer};

    #[test]
    fn test_new() {
        let buffer = CircularBuffer::new(5);
        assert_eq!(buffer.get_buffer(), vec![0.0; 5]);
    }

    #[test]
    fn test_add_chunk_fit_exactly() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buffer.get_buffer(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_chunk_smaller_than_buffer() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0]);
        assert_eq!(buffer.get_buffer(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_add_chunk_wrap_around() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0]);
        buffer.add_chunk(&[4.0, 5.0, 6.0]);
        // Expected: Last 5 values, wrapped around
        assert_eq!(buffer.get_buffer(), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_add_chunk_larger_than_buffer() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        // Expected: Only the last 5 values are retained
        assert_eq!(buffer.get_buffer(), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_copy_to_buffer_exact_size() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut output = vec![0.0; 5];
        buffer.copy_to_buffer(&mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_copy_to_buffer_wrap_around() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0]);
        buffer.add_chunk(&[4.0, 5.0, 6.0]);

        let mut output = vec![0.0; 5];
        buffer.copy_to_buffer(&mut output);
        assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Output slice size mismatch")]
    fn test_copy_to_buffer_invalid_output_size() {
        let mut buffer = CircularBuffer::new(5);
        buffer.add_chunk(&[1.0, 2.0, 3.0]);

        let mut output = vec![0.0; 4]; // Incorrect size
        buffer.copy_to_buffer(&mut output);
    }

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
