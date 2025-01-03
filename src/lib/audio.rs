use crate::filters::new_lowpass_1;
use crate::tuner::{identify_frequency, Resonators};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::time::{Duration, Instant};
use std::{
    sync::{Arc, Mutex},
    thread,
};

pub trait Block {
    type InputData;
    type OutputData;

    fn set_source(&mut self, receiver: Receiver<Self::InputData>);
    fn add_output(&mut self, sender: Sender<Self::OutputData>);
    fn pipe_output_to<B>(&mut self, downstream: &mut B)
    where
        B: Block<InputData = Self::OutputData>,
    {
        let (sender, receiver) = unbounded();
        self.add_output(sender);
        downstream.set_source(receiver);
    }
}

/// Represents a detected pulse in the input signal.
#[derive(Clone, Copy)]
pub struct Pulse {
    pub energy: f32,
    pub duration: usize,
    pub max_amplitude: f32,
    pub max_time_index: usize,
}

pub struct PulseDetectorBlock {
    pub threshold: f32,
    source: Option<Receiver<(u128, Arc<Vec<f32>>)>>,
    output: Vec<Sender<Pulse>>,
}

pub struct InputBlock {
    pub device_name: String,
    pub device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    output: Vec<Sender<(u128, Arc<Vec<f32>>)>>,
}

impl Block for InputBlock {
    type InputData = ();
    type OutputData = (u128, Arc<Vec<f32>>);

    fn set_source(&mut self, _receiver: Receiver<Self::InputData>) {}

    fn add_output(&mut self, sender: Sender<Self::OutputData>) {
        self.output.push(sender);
    }
}

impl InputBlock {
    pub fn new(device_name: &str) -> Self {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .unwrap()
            .find(|d| d.name().unwrap_or_default() == device_name)
            .unwrap();
        let config = device.default_input_config().unwrap();

        Self {
            device_name: device_name.into(),
            device,
            config,
            output: vec![],
        }
    }

    pub fn run(&self) {
        let device = self.device.clone();
        let outputs = self.output.clone();
        let config = self.config.clone();

        thread::spawn(move || {
            let channels = config.channels() as usize;

            let now = Instant::now();

            let stream = device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &_| {
                        let timestamp = Instant::now().duration_since(now).as_nanos();
                        let mono_chunk: Vec<f32> = data
                            .chunks(channels)
                            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                            .collect();

                        let buffer = Arc::new(mono_chunk);
                        for listener in &outputs {
                            listener.send((timestamp, buffer.clone())).unwrap();
                        }
                    },
                    |err| eprintln!("Stream error: {}", err),
                    None,
                )
                .unwrap();

            stream.play().unwrap();
        });
    }
}

impl Block for PulseDetectorBlock {
    type InputData = (u128, Arc<Vec<f32>>);
    type OutputData = Pulse;

    fn set_source(&mut self, receiver: Receiver<Self::InputData>) {
        self.source = Some(receiver);
    }

    fn add_output(&mut self, sender: Sender<Self::OutputData>) {
        self.output.push(sender);
    }
}

impl PulseDetectorBlock {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            source: None,
            output: vec![],
        }
    }

    pub fn run(&self) {
        if let Some(source) = self.source.clone() {
            let threshold = self.threshold;
            let listeners = self.output.clone();

            thread::spawn(move || {
                let mut detector = PulseDetector::new(threshold);
                while let Ok((_timestamp, sample)) = source.recv() {
                    if let Some((pulse, _start)) = detector.process_new_samples(&sample) {
                        for listener in &listeners {
                            listener.send(pulse).unwrap();
                        }
                    }
                }
            });
        }
    }
}

/// Detects pulses in an audio signal based on energy thresholding.
pub struct PulseDetector {
    threshold: f32,
    progress: Option<Pulse>,
}

impl PulseDetector {
    /// Creates a new pulse detector.
    ///
    /// # Arguments
    /// * `sample_rate` - The sampling rate of the input signal in Hz.
    /// * `cutoff_frequency` - Cutoff frequency for the lowpass filter in Hz.
    /// * `threshold` - Energy threshold for detecting pulses.
    pub fn new(threshold: f32) -> Self {
        PulseDetector {
            threshold,
            progress: None,
        }
    }

    /// Processes a new batch of samples and detects pulses based on energy.
    ///
    /// The function uses a lowpass filter to smooth the squared amplitude (energy) of the signal.
    /// Pulses are detected when the energy exceeds a specified threshold. It handles different
    /// scenarios, including pulses that start in the current input, continue from previous samples,
    /// or end within the input.
    ///
    /// # Arguments
    /// * `samples` - A slice of input samples (assumed to be pre-processed).
    ///
    /// # Returns
    /// * `None` - If no complete pulse is detected.
    /// * `Some((Pulse, usize))` - If a pulse is completed, returns the `Pulse` and its
    ///    end index within the batch.
    ///
    /// # Scenarios
    /// 1. **No Pulse**:
    ///    - The energy of all samples is below the threshold.
    ///    - The function will return `None` without modifying the internal state.
    ///
    /// 2. **Input contains a full pulse**:
    ///    - A pulse begins in the current batch (energy exceeds the threshold), but the
    ///      energy never falls back below the threshold within the same batch.
    ///    - The function will update `self.progress` to track the ongoing pulse and return `None`.
    ///
    /// 3. **Pulse starts in the current input sample but not completed**:
    ///    - A pulse begins and ends entirely within the current batch.
    ///    - The function will return the completed `Pulse` and its end index.
    ///
    /// 4. **Pulse started from a previous sample**:
    ///    - The function resumes tracking a pulse started in a previous batch (`self.progress`
    ///      is already set).
    ///    - If the pulse ends in the current batch, it will return the completed `Pulse`.
    ///    - If the pulse does not end, the function will update `self.progress` and return `None`.
    pub fn process_new_samples(&mut self, samples: &[f32]) -> Option<(Pulse, usize)> {
        let mut start = 0;

        // If no pulse is in progress, look for the start of a new pulse
        if self.progress.is_none() {
            for (i, &y) in samples.iter().enumerate() {
                if y > self.threshold {
                    self.progress = Some(Pulse {
                        energy: y,
                        duration: 0,
                        max_amplitude: y,
                        max_time_index: 0,
                    });
                    start = i + 1; // Start processing the next samples
                    break;
                }
            }
        }

        if let Some(mut progress) = self.progress.take() {
            for (i, &y) in samples.iter().enumerate().skip(start) {
                progress.energy += y;
                progress.duration += 1;

                // End the pulse if energy falls below the threshold
                if y < self.threshold {
                    self.progress = None;
                    return Some((progress, i));
                }

                // Update max amplitude if necessary
                if y > progress.max_amplitude {
                    progress.max_amplitude = y;
                    progress.max_time_index = progress.duration;
                }
            }
            // If the pulse is still in progress, reassign it back to self.progress
            self.progress = Some(progress);
        }

        None
    }
}

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
            self.buffer.copy_from_slice(&chunk[start..]);
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

impl Block for LowpassBlock {
    type InputData = (u128, Arc<Vec<f32>>);
    type OutputData = (u128, Arc<Vec<f32>>);

    fn set_source(&mut self, receiver: Receiver<Self::InputData>) {
        self.source = Some(receiver);
    }

    fn add_output(&mut self, sender: Sender<Self::OutputData>) {
        self.output.push(sender);
    }
}

pub struct LowpassBlock {
    pub cutoff_frequency: f32,
    pub sample_rate: usize,
    pub transform: Option<Arc<dyn Fn(f32) -> f32 + Send + Sync>>,
    source: Option<Receiver<(u128, Arc<Vec<f32>>)>>,
    output: Vec<Sender<(u128, Arc<Vec<f32>>)>>,
}

impl LowpassBlock {
    pub fn new(
        cutoff_frequency: f32,
        sample_rate: usize,
        transform: Option<impl Fn(f32) -> f32 + Send + Sync + 'static>,
    ) -> Self {
        Self {
            cutoff_frequency,
            sample_rate,
            transform: transform.map(|f| Arc::new(f) as Arc<dyn Fn(f32) -> f32 + Send + Sync>),
            source: None,
            output: vec![],
        }
    }

    pub fn run(&self) {
        if let Some(source) = self.source.clone() {
            let sample_rate = self.sample_rate;
            let cutoff_frequency = self.cutoff_frequency;

            let listeners = self.output.clone();
            let transform = self.transform.clone();

            thread::spawn(move || {
                let mut filter = new_lowpass_1(cutoff_frequency, sample_rate as f32);
                while let Ok((timestamp, packet)) = source.recv() {
                    let mut buffer = vec![0.0; packet.len()];
                    for (i, &x) in packet.iter().enumerate() {
                        buffer[i] = filter.apply(if let Some(f) = transform.as_ref() {
                            f(x)
                        } else {
                            x
                        });
                    }

                    let buffer = Arc::new(buffer);

                    for listener in &listeners {
                        listener.send((timestamp, Arc::clone(&buffer))).unwrap();
                    }
                }
            });
        }
    }
}

impl Block for PitchDetectorBlock {
    type InputData = (u128, Arc<Vec<f32>>);
    type OutputData = Option<f32>;

    fn set_source(&mut self, receiver: Receiver<Self::InputData>) {
        self.source = Some(receiver);
    }

    fn add_output(&mut self, sender: Sender<Self::OutputData>) {
        self.output.push(sender);
    }
}

pub struct PitchDetectorBlock {
    pub candidate_frequencies: Vec<f32>,
    pub sample_rate: usize,
    pub window_size: usize,
    pub update_interval: u64,
    source: Option<Receiver<(u128, Arc<Vec<f32>>)>>,
    output: Vec<Sender<Option<f32>>>,
}

impl PitchDetectorBlock {
    pub fn new(
        candidate_frequencies: Vec<f32>,
        sample_rate: usize,
        window_size: usize,
        update_interval: u64,
    ) -> Self {
        Self {
            candidate_frequencies,
            sample_rate,
            window_size,
            update_interval,
            source: None,
            output: vec![],
        }
    }

    pub fn run(&self) {
        if let Some(source) = self.source.clone() {
            let sample_rate = self.sample_rate;
            let candidate_frequencies = self.candidate_frequencies.clone();

            let buffer = Arc::new(Mutex::new(CircularBuffer::new(self.window_size)));
            let buffer_reader = Arc::clone(&buffer);

            let cutoff_frequency = 10.0;
            let resonators = Arc::new(Mutex::new(Resonators::new(
                &candidate_frequencies,
                sample_rate,
                cutoff_frequency,
                sample_rate / 20,
            )));
            let resonators_reader = Arc::clone(&resonators);

            thread::spawn(move || {
                while let Ok((_timestamp, packet)) = source.recv() {
                    buffer.lock().unwrap().add_chunk(&packet);
                    resonators.lock().unwrap().process_new_samples(&packet);
                }
            });

            let update_interval = self.update_interval;
            let listeners = self.output.clone();
            let max_window = self.window_size;

            thread::spawn(move || {
                let mut work_buffer = vec![0.0; max_window];
                loop {
                    let (f_candidate, _) = resonators_reader.lock().unwrap().current_peak();
                    buffer_reader
                        .lock()
                        .unwrap()
                        .copy_to_buffer(&mut work_buffer);

                    let mut pitch = identify_frequency(
                        &work_buffer,
                        sample_rate,
                        f_candidate - 5.0,
                        f_candidate + 5.0,
                        true,
                    );

                    // Sometimes the 2nd harmonic is more energetic than the fundamental frequency
                    // but it's not enough to trigger detection via autocorrelation.  So we will
                    // also check the half-frequency to see if it's a match
                    if pitch.is_none() {
                        pitch = identify_frequency(
                            &work_buffer,
                            sample_rate,
                            0.5 * f_candidate - 5.0,
                            0.5 * f_candidate + 5.0,
                            true,
                        );
                    }

                    for listener in &listeners {
                        listener.send(pitch).unwrap();
                    }

                    thread::sleep(Duration::from_millis(update_interval));
                }
            });
        }
    }
}

pub fn get_input_device(device_name: String) -> Option<cpal::Device> {
    let host = cpal::default_host();
    host.input_devices()
        .unwrap()
        .find(|d| d.name().unwrap_or_default() == device_name)
}

/// Start the audio processing loop
pub fn start_audio_processing(device: cpal::Device, output: Vec<Sender<(u128, Arc<Vec<f32>>)>>) {
    thread::spawn(move || {
        let config = device.default_input_config().unwrap();
        let channels = config.channels() as usize;

        let now = Instant::now();

        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &_| {
                    let timestamp = Instant::now().duration_since(now).as_nanos();
                    let mono_chunk: Vec<f32> = data
                        .chunks(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect();

                    let buffer = Arc::new(mono_chunk);
                    for listener in &output {
                        listener.send((timestamp, buffer.clone())).unwrap();
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )
            .unwrap();

        stream.play().unwrap();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

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

    #[test]
    fn test_single_pulse() {
        let sample_rate = 44100;
        let cutoff_frequency = sample_rate as f32;
        let threshold = 0.01;

        // Simulated signal with one pulse
        let samples = vec![
            0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0, // Single pulse
        ];

        let mut filter = new_lowpass_1(cutoff_frequency, sample_rate as f32);
        let mut energy = samples.clone();
        for (i, &x) in samples.iter().enumerate() {
            energy[i] = filter.apply(x * x);
        }

        let mut detector = PulseDetector::new(threshold);
        if let Some((pulse, _)) = detector.process_new_samples(&energy) {
            assert!(pulse.energy > 0.1, "Energy should be above threshold");
            assert_eq!(pulse.duration, 6, "Pulse duration should be 6 samples");
            assert_approx_eq!(pulse.max_amplitude, 0.8930, 1e-3);
        } else {
            panic!("No pulse detected when one was expected.");
        }
    }

    #[test]
    fn test_multiple_pulses() {
        let sample_rate = 44100;
        let cutoff_frequency = sample_rate as f32;
        let threshold = 0.01;

        let mut detector = PulseDetector::new(threshold);

        // Simulated signal with one pulse
        let samples = vec![
            0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0, // first pulse
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // filler
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // filler
            0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0, // second pulse
        ];

        let mut filter = new_lowpass_1(cutoff_frequency, sample_rate as f32);
        let mut energy = samples.clone();
        for (i, &x) in samples.iter().enumerate() {
            energy[i] = filter.apply(x * x);
        }

        // Detect the first pulse
        if let Some((pulse, end_index)) = detector.process_new_samples(&energy) {
            assert!(pulse.energy > 0.1, "Energy should be above threshold");
            assert_eq!(
                pulse.duration, 6,
                "First pulse duration should be 6 samples"
            );
            assert_approx_eq!(pulse.max_amplitude, 0.8930, 1e-3);

            // Detect the second pulse
            if let Some((pulse, _)) = detector.process_new_samples(&energy[end_index + 1..]) {
                assert!(pulse.energy > 0.1, "Energy should be above threshold");
                assert_eq!(
                    pulse.duration, 6,
                    "Second pulse duration should be 7 samples"
                );
                assert_approx_eq!(pulse.max_amplitude, 0.8930, 1e-3);
            } else {
                panic!("Second pulse not detected.");
            }
        } else {
            panic!("First pulse not detected.");
        }
    }
}
