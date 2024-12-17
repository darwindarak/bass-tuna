use crate::tuner::identify_frequency;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::Sender;
use std::{
    sync::{Arc, Mutex},
    thread,
};

/// A fixed-size circular buffer
/// - `buffer`: The underlying storage for the circular buffer.
/// - `max_size`: The maximum capacity of the buffer.
/// - `write_index`: The position in the buffer where the next chunk will be written.
pub struct CircularBuffer {
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
    pub fn new(max_size: usize) -> Self {
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
    pub fn add_chunk(&mut self, chunk: &[f32]) {
        if chunk.len() < self.max_size {
            // Case 1: Chunk can fit into the buffer without overflow
            let remaining = self.max_size - self.write_index;

            if remaining >= chunk.len() {
                self.buffer[self.write_index..(self.write_index + chunk.len())]
                    .copy_from_slice(chunk);
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
    pub fn copy_to_buffer(&self, output: &mut [f32]) {
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
    pub fn get_buffer(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.max_size);
        result.extend_from_slice(&self.buffer[self.write_index..]);
        result.extend_from_slice(&self.buffer[..self.write_index]);
        result
    }
}

/// Start the audio processing loop
pub fn start_audio_processing(device_name: String, pitch_tx: Sender<Option<f32>>) {
    thread::spawn(move || {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .unwrap()
            .find(|d| d.name().unwrap_or_default() == device_name)
            .expect("Device not found");

        let config = device.default_input_config().unwrap();
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let max_window = sample_rate / 10; // 2 seconds

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
                let pitch = identify_frequency(&work_buffer, sample_rate as f32, 30.0, 150.0);

                pitch_tx.send(pitch).ok();
                thread::sleep(std::time::Duration::from_millis(150)); // Control processing frequency
            }
        });

        let input_buffer = Arc::clone(&circular_buffer);
        let stream = device
            .build_input_stream(
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
            )
            .unwrap();

        stream.play().unwrap();
    });
}
