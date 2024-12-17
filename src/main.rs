mod audio;
mod tui;
mod tuner;

use crate::tui::{update, view, Message, Model};
use ratatui::crossterm::event::{self, Event, KeyCode};
use std::io::{self};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    let (pitch_tx, pitch_rx): (Sender<Option<f32>>, Receiver<Option<f32>>) = channel();
    let mut model = Model::new();

    loop {
        terminal.draw(|f| view(f, &model)).ok();

        if event::poll(Duration::from_millis(100))? {
            if let Ok(event) = event::read() {
                match event {
                    Event::Key(key_event) => match key_event.code {
                        KeyCode::Up => update(&mut model, Message::SelectPreviousDevice, &pitch_tx),
                        KeyCode::Down => update(&mut model, Message::SelectNextDevice, &pitch_tx),
                        KeyCode::Enter => update(&mut model, Message::ConfirmDevice, &pitch_tx),
                        KeyCode::Esc => update(&mut model, Message::Exit, &pitch_tx),
                        KeyCode::Char('q') => update(&mut model, Message::Exit, &pitch_tx),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        if let Ok(pitch) = pitch_rx.try_recv() {
            update(&mut model, Message::UpdatePitch(pitch), &pitch_tx);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::audio::CircularBuffer;
    use crate::tuner::identify_frequency;

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
