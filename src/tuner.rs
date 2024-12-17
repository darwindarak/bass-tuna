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
