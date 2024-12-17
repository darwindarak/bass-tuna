use realfft::RealFftPlanner;
use std::io::{self, Write};

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
    fn copy_to_buffer(&mut self, output: &mut [f32]) {
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

fn harmonic_energy(f_amplitudes: &[f32], freq_bin: usize, n_harmonics: usize) -> f32 {
    let mut energy = 0.0;
    let window = f_amplitudes.len();

    for n in 1..=n_harmonics {
        let bin = n * freq_bin;
        if bin < window {
            energy += f_amplitudes[bin].abs();
        }
    }

    energy
}

fn freq_to_bin(freq: f32, window: usize, sample_rate: f32) -> usize {
    (freq * window as f32 / sample_rate).round() as usize
}

fn identify_pitch(
    input: &[f32],
    options: &[f32],
    sample_rate: f32,
    n_harmonics: usize,
    f_range: f32,
) -> (f32, usize, f32) {
    // Perform FFT
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(input.len());

    let mut scratch = fft.make_input_vec();
    scratch.copy_from_slice(input);
    let mut output = fft.make_output_vec();

    fft.process(&mut scratch, &mut output)
        .expect("should be ok");

    let amplitudes: Vec<f32> = output.iter().map(|c| c.norm()).collect();
    let window = input.len();

    // Convert options to bins
    let options_bins: Vec<usize> = options
        .iter()
        .map(|&freq| freq_to_bin(freq, window, sample_rate))
        .collect();

    // Calculate harmonic energies
    let energies: Vec<f32> = options_bins
        .iter()
        .map(|&bin| harmonic_energy(&amplitudes, bin, n_harmonics))
        .collect();

    let i = energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    // Refine pitch
    let resolution = sample_rate / window as f32;
    let bin_width = (f_range / resolution).ceil() as usize;

    let center_bin = freq_to_bin(options[i], window, sample_rate);
    let bins = (center_bin.saturating_sub(bin_width))..=(center_bin + bin_width);

    let energies_hires: Vec<f32> = bins
        .clone()
        .map(|bin| harmonic_energy(&amplitudes, bin, n_harmonics))
        .collect();

    let j = energies_hires
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    (
        options[i],
        i,
        (*bins.start() + j) as f32 * sample_rate / window as f32,
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read a WAV file
    print!("Enter the path to the WAV file: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let path = input.trim();

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f32;
    let channels = spec.channels as usize;

    println!(
        "Reading WAV file with sample rate: {} Hz, channels: {}",
        sample_rate, channels
    );

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32) // Normalize to [-1.0, 1.0]
        .collect();

    // Convert to mono if necessary
    let mono_samples: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    let max_window = 2 * 88_200; // 2 seconds of data at 44.1 kHz
    let chunk_size = 1024;
    let options = vec![41.2, 55.0, 73.4, 98.0]; // E1, A1, D2, G2 frequencies

    println!("Processing the WAV file...");

    let mut start = 0;
    while start + max_window <= mono_samples.len() {
        let work_buffer = &mono_samples[start..start + max_window];
        let (_, _, pitch) = identify_pitch(work_buffer, &options, sample_rate, 5, 5.0);
        println!("Detected pitch: {:.4} Hz", pitch);
        start += chunk_size;
    }

    println!("Processing complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CircularBuffer;

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
}
