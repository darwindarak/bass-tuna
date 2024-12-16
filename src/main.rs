use realfft::RealFftPlanner;
use std::io::{self, Write};

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
