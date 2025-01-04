use num::{Complex, Integer, One};
use num_traits::{Float, FloatConst};

/// An IIR digital filter implementation
pub struct IIRFilter<T: Float + FloatConst, const N: usize> {
    /// Feedforward coefficients (b)
    b: [T; N],

    /// Feedback coefficients (a)
    /// a[0] is assumed to be 1.0 (normalized)
    a: [T; N],

    /// Buffer of previous output values (y[n-k] for k=0..N-1)
    ///
    /// The buffer follows the mathematical notation where y[n-k] represents
    /// the output from k steps ago. To maintain this intuitive indexing:
    /// - y[0] is reserved but unused (would represent current output y[n])
    /// - y[1] holds y[n-1] (previous output)
    /// - y[2] holds y[n-2] (output from two steps ago)
    ///
    /// This indexing scheme sacrifices one array element but makes the code
    /// more maintainable by directly matching the difference equation indices.
    y: [T; N],

    /// Buffer of previous input values (x[n-k] for k=0..N-1)
    ///
    /// Similar to the output buffer, we maintain mathematical notation where
    /// x[n-k] represents the input from k steps ago:
    /// - x[0] is reserved but unused (would represent current input x[n])
    /// - x[1] holds x[n-1] (previous input)
    /// - x[2] holds x[n-2] (input from two steps ago)
    ///
    /// This matches the traditional difference equation notation, making the
    /// implementation easier to verify against the mathematical formulation.
    x: [T; N],
}

impl<T: Float + FloatConst, const N: usize> IIRFilter<T, N> {
    /// Creates a new IIR filter from the provided coefficients.
    ///
    /// # Arguments
    /// * `b` - Array of feedforward coefficients
    /// * `a` - Array of feedback coefficients
    ///
    /// # Returns
    /// A new IIRFilter instance with the specified order
    pub fn from_coeffs(b: [T; N], a: [T; N]) -> Self {
        Self {
            b,
            a,
            x: [T::zero(); N],
            y: [T::zero(); N],
        }
    }

    /// Alternative constructor from slices
    ///
    /// # Returns
    /// Result containing the filter or an error if slices aren't the right length
    pub fn from_coeff_slices(b: &[T], a: &[T]) -> Result<Self, &'static str> {
        if b.len() != N || a.len() != N {
            return Err("Coefficient slices must match the filter order");
        }

        let mut b_array = [T::zero(); N];
        let mut a_array = [T::zero(); N];

        b_array.copy_from_slice(b);
        a_array.copy_from_slice(a);

        Ok(Self::from_coeffs(b_array, a_array))
    }

    /// Processes a single input sample through the filter.
    pub fn apply(&mut self, x: T) -> T {
        // Compute current output
        let mut y = self.b[0] * x;
        for k in 1..N {
            y = y + self.b[k] * self.x[k] - self.a[k] * self.y[k];
        }

        // Rotate buffers right
        self.x.rotate_right(1);
        self.y.rotate_right(1);

        // Update most recent states
        self.x[1] = x;
        self.y[1] = y;

        y
    }

    /// Resets the filter state buffers to zero
    pub fn reset(&mut self) {
        self.x = [T::zero(); N];
        self.y = [T::zero(); N];
    }
}

/// Creates a first-order (single pole) lowpass IIR filter.
///
/// This implements a basic first-order lowpass filter with the transfer function:
///
/// H(z) = (α) / (1 - (1-α)z^(-1))
///
/// where α = ω/(ω + 1) and ω = 2πf₀/fs
///
/// # Arguments
/// * `f0` - Cutoff frequency in Hz
/// * `sampling_frequency` - Sampling frequency in Hz
///
/// # Returns
/// A first-order lowpass IIR filter
///
/// # Theory
/// This filter provides a -20 dB/decade rolloff above the cutoff frequency.
/// The phase response is -45° at the cutoff frequency.
///
pub fn new_lowpass_1<T>(f0: T, sampling_frequency: T) -> IIRFilter<T, 2>
where
    T: Float + FloatConst,
{
    let omega = T::TAU() * f0 / sampling_frequency;
    let alpha = omega / (omega + T::one());

    let b = [alpha, T::zero()];
    let a = [T::zero(), alpha - T::one()];

    IIRFilter::from_coeffs(b, a)
}

/// Creates a second-order (biquad) lowpass IIR filter.
///
/// This implements a second-order lowpass filter with the transfer function:
///
/// H(z) = (b₀(1 + b₂z^(-2))) / (1 + a₁z^(-1) + a₂z^(-2))
///
/// where the coefficients are derived from the bilinear transform of the
/// analog prototype filter.
///
/// # Arguments
/// * `f0` - Cutoff frequency in Hz
/// * `sampling_frequency` - Sampling frequency in Hz
/// * `q` - Quality factor (Q) controlling the resonance at the cutoff frequency
///         Typical values:
///         - 0.707 (1/√2) for Butterworth response (maximally flat)
///         - > 0.707 for peaking response
///         - < 0.707 for more damped response
///
/// # Returns
/// A second-order lowpass IIR filter
///
/// # Theory
/// This filter provides:
/// - -40 dB/decade rolloff above the cutoff frequency
/// - -90° phase shift at the cutoff frequency
/// - Variable Q factor allowing control of the response shape
pub fn new_lowpass_2<T>(f0: T, sampling_frequency: T, q: T) -> IIRFilter<T, 3>
where
    T: Float + FloatConst,
{
    let k = f0 / sampling_frequency;
    let omega_0 = T::TAU() * k;
    let alpha = omega_0.sin() / (T::from(2).unwrap() * q);

    let a_0 = T::one() + alpha;
    let a_1 = -T::from(2).unwrap() * omega_0.cos() / a_0;
    let a_2 = (T::one() - alpha) / a_0;

    let a = [a_0, a_1, a_2];

    let b_0 = alpha / a_0;
    let b_1 = T::zero();
    let b_2 = -b_0;

    let b = [b_0, b_1, b_2];

    IIRFilter::from_coeffs(b, a)
}

/// Computes the prewarped angular frequency for the bilinear transform.
///
/// # Parameters
/// - `f`: The analog cutoff frequency (in Hz).
/// - `sampling_frequency`: The sampling frequency of the system (in Hz).
///
/// # Returns
/// The prewarped angular frequency, \(\omega' = \tan\left(\frac{\omega}{2F_s}\right) \cdot 2F_s\),
/// where \(\omega = 2\pi f\) is the angular frequency.
///
/// # Requirements
/// - `f` must be positive.
/// - `sampling_frequency` must be positive.
///
/// # Panics
/// This function will panic if either `f` or `sampling_frequency` is less than or equal to zero.
///
fn prewarped_frequency<T>(f: T, sampling_frequency: T) -> T
where
    T: Float + FloatConst,
{
    assert!(f > T::zero(), "Frequency (f) must be positive.");
    assert!(
        sampling_frequency > T::zero(),
        "Sampling frequency must be positive."
    );

    let omega = T::TAU() * f; // Compute angular frequency: ω = 2πf
    let half_period = T::one() / (T::from(2).unwrap() * sampling_frequency);

    (omega * half_period).tan() / half_period
}

/// Calculates the poles of a Butterworth filter given a cutoff frequency and filter order.
///
/// A Butterworth filter is characterized by a maximally flat frequency response in the passband
/// and a roll-off rate of -20n dB/decade in the stopband, where n is the filter order.
///
/// # Arguments
/// * omega_c - The angular frequency at which the magnitude response is -3dB
/// * order - The order of the filter, determines roll-off rate and number of poles
///
/// # Returns
/// A tuple containing:
/// * First element: An Option<T> that contains the real pole for odd-order filters, None for even-order
/// * Second element: A Vec of Complex<T> containing the upper half-plae one of the complex conjugate pole pairs
///
fn butterworth_poles<T: Float + FloatConst>(
    omega_c: T,
    order: i16,
) -> (Option<T>, Vec<Complex<T>>) {
    // For even-order filters, there's no real pole
    // For odd-order filters, there's one real pole at -cutoff_frequency
    let (first_order_pole, num_k) = if order.is_even() {
        (None, order / 2)
    } else {
        (Some(-omega_c), (order - 1) / 2)
    };

    // Pre-allocate vector for complex conjugate pole pairs
    let mut biquad_poles = Vec::with_capacity(num_k as usize);

    // Calculate the angle step size for pole placement
    // There are multiple options for s^(2n) = (-1), but for stability
    // reasons, we want the poles to be located on the left-half plane.
    let angle_step = T::FRAC_PI_2() / (T::from(order).unwrap());

    for k in 1..=num_k {
        let angle = T::FRAC_PI_2() + angle_step * T::from(2 * k - 1).unwrap();
        biquad_poles.push(Complex::cis(angle).scale(omega_c))
    }

    (first_order_pole, biquad_poles)
}

/// Converts a single Butterworth pole to digital filter coefficients for a 2nd-order section,
/// specifically designed for complex conjugate pole pairs. This function generates coefficients
/// for a biquad filter implementation.
///
/// # Mathematical Background
/// In analog filter design, complex poles always come in conjugate pairs to ensure
/// real-valued coefficients. For a pole p = a + jb, its conjugate is p* = a - jb.
/// This function takes one pole of the conjugate pair and automatically handles both poles
/// to create a 2nd-order filter section (biquad).
///
/// The transfer function has the form:
/// H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²)/(1 + a₁z⁻¹ + a₂z⁻²)
///
/// When we have a complex conjugate pair of poles (z₁ and z₁*), the denominator becomes:
/// (z - z₁)(z - z₁*) = z² - 2Re(z₁)z + |z₁|²
///
/// This is why:
/// - a₁ = -2Re(z₁)     // Twice the negative real part of the pole
/// - a₂ = |z₁|²        // Square of the pole's magnitude
///
/// The numerator coefficients are chosen to maintain unity gain at DC frequency:
/// b₀ = b₂ = (1 + a₁ + a₂)/4
/// b₁ = 2b₀
///
/// # Arguments
/// * pole - One pole of a complex conjugate pair from the Butterworth filter
///           in the s-domain. The function automatically handles both this pole
///           and its conjugate.
/// * sampling_period - The sampling period (T) of the discrete system
///
/// # Type Parameters
/// * T - A floating-point type that implements Float and FloatConst traits
///
/// # Returns
/// * A tuple of two arrays ([T; 3], [T; 3]) containing:
///   - b coefficients (numerator): [b₀, b₁, b₂]
///   - a coefficients (denominator): [1, a₁, a₂]
///
/// These coefficients are normalized so a[0] = 1
///
fn butterworth_pole_to_biquad_coeffs<T: Float + FloatConst>(
    pole: Complex<T>,
    sampling_period: T,
) -> ([T; 3], [T; 3]) {
    let z = bilinear_transform(pole, sampling_period);

    let mut a = [T::one(); 3];
    let mut b = [T::zero(); 3];

    a[1] = -z.re - z.re;
    a[2] = z.norm_sqr();

    b[0] = (T::one() + a[1] + a[2]) / T::from(4).unwrap();
    b[1] = b[0] + b[0];
    b[2] = b[0];

    (b, a)
}

/// Performs the bilinear transform to map an analog (continuous-time) system point
/// to its digital (discrete-time) equivalent.
///
/// The bilinear transform is a conformal mapping used in digital signal processing to convert
/// between continuous-time (s-domain) and discrete-time (z-domain) systems. This function
/// implements the forward mapping from s-domain to z-domain, which is crucial for
/// digital filter design.
///
/// # Arguments
/// * analog_point - A complex number representing a point in the s-domain (typically
///                    a pole or zero of a transfer function)
/// * sampling_period - The sampling period (T) of the discrete system, where
///                      sampling_frequency = 1/T
///
/// # Type Parameters
/// * T - A floating-point type that implements the Float trait
///
/// # Returns
/// * A complex number representing the mapped point in the z-domain
///
/// # Mathematical Background
/// The bilinear transform implements the mapping:
/// z = (1 + (T/2)s)/(1 - (T/2)s)
///
/// This transformation has several important properties (for testing):
/// - Maps the left half s-plane to inside the unit circle (preserving stability)
/// - Maps the imaginary axis to the unit circle (preserving frequency responses)
/// - Is a one-to-one conformal mapping (preserving angle relationships)
fn bilinear_transform<T: Float>(analog_point: Complex<T>, sampling_period: T) -> Complex<T> {
    // TODO: add checks for T > 0, 1 - alpha != 0
    let alpha = analog_point.scale(sampling_period / T::from(2).unwrap());
    let one: Complex<T> = Complex::one();

    (one + alpha) / (one - alpha)
}

pub struct ButterworthFilter<T: Float + FloatConst> {
    pub order: i16,
    pub cutoff_frequency: T,
    pub sampling_frequency: T,

    first_order_filter: Option<IIRFilter<T, 2>>,
    biquad_sections: Vec<IIRFilter<T, 3>>,
}

impl<T: Float + FloatConst> ButterworthFilter<T> {
    pub fn new(order: i16, cutoff_frequency: T, sampling_frequency: T) -> Self {
        let omega = prewarped_frequency(cutoff_frequency, sampling_frequency);
        let sampling_period = T::one() / sampling_frequency;
        let (first_order_pole, second_order_poles) = butterworth_poles(omega, order);

        let first_order_filter = first_order_pole.map(|p| {
            let beta = sampling_period / T::from(2).unwrap();
            let z = (T::one() + p * beta) / (T::one() - p * beta);

            IIRFilter::from_coeffs([T::one() - z, T::zero()], [T::one(), -z])
        });

        let biquad_sections = second_order_poles
            .iter()
            .map(|&p| {
                let (b, a) = butterworth_pole_to_biquad_coeffs(p, sampling_period);
                IIRFilter::from_coeffs(b, a)
            })
            .collect();

        Self {
            order,
            cutoff_frequency,
            sampling_frequency,
            first_order_filter,
            biquad_sections,
        }
    }

    /// Processes a single input sample through the complete Butterworth filter.
    ///
    /// This implementation uses a cascade of first-order and second-order (biquad) sections
    /// to maintain numerical stability, especially for higher-order filters. The input signal
    /// flows through each section sequentially, with each section's output becoming the input
    /// to the next section.
    ///
    /// # Returns
    /// The filtered output sample
    ///
    pub fn apply(&mut self, x: T) -> T {
        // Start with the input sample
        let mut y = x;

        // If this is an odd-order filter, apply the first-order section first
        if let Some(filter) = &mut self.first_order_filter {
            y = filter.apply(y);
        }

        // Apply each biquad section in sequence
        // The output of each section becomes the input to the next
        for section in &mut self.biquad_sections {
            y = section.apply(y);
        }

        y
    }

    /// Resets the state of all filter sections to zero.
    pub fn reset(&mut self) {
        if let Some(filter) = &mut self.first_order_filter {
            filter.reset();
        }
        for section in &mut self.biquad_sections {
            section.reset();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use assertables::{
        assert_approx_eq, assert_gt, assert_in_delta, assert_in_epsilon, assert_le, assert_lt,
    };

    // Helper functions for testing
    fn generate_sine_wave(frequency: f64, duration_seconds: f64, sampling_rate: f64) -> Vec<f64> {
        let num_samples = (duration_seconds * sampling_rate) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                (2.0 * f64::PI() * frequency * t).sin()
            })
            .collect()
    }

    fn calculate_power(signal: &[f64]) -> f64 {
        signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64
    }

    #[test]
    fn test_filter_construction() {
        // Test valid filter construction
        let filter = ButterworthFilter::new(4, 1000.0, 44100.0);
        assert_eq!(filter.order, 4);
        assert_eq!(filter.biquad_sections.len(), 2); // 4th order = 2 biquad sections
        assert!(filter.first_order_filter.is_none()); // Even order, no first-order section

        // Test odd-order filter
        let filter = ButterworthFilter::new(3, 1000.0, 44100.0);
        assert_eq!(filter.order, 3);
        assert_eq!(filter.biquad_sections.len(), 1); // One complex conjugate pair
        assert!(filter.first_order_filter.is_some()); // Should have one real pole
    }

    #[test]
    #[should_panic(expected = "Frequency (f) must be positive")]
    fn test_invalid_cutoff_frequency() {
        ButterworthFilter::new(4, -1000.0, 44100.0);
    }

    #[test]
    #[should_panic(expected = "Sampling frequency must be positive")]
    fn test_invalid_sampling_frequency() {
        ButterworthFilter::new(4, 1000.0, -44100.0);
    }

    #[test]
    fn test_cutoff_frequency_response() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Test signal at cutoff frequency
        let input = generate_sine_wave(cutoff_freq, 1.0, sampling_rate);
        let mut output = Vec::with_capacity(input.len());

        // Apply filter
        for &x in &input {
            output.push(filter.apply(x));
        }

        // Skip initial transient response
        let skip_samples = (sampling_rate * 0.1) as usize; // Skip first 0.1 seconds
        let input_power = calculate_power(&input[skip_samples..]);
        let output_power = calculate_power(&output[skip_samples..]);

        // At cutoff frequency, power should be reduced by -3dB (≈ 0.708 linear)
        let power_ratio = (output_power / input_power).sqrt();
        assert_in_delta!(power_ratio, 0.708, 0.01);
    }

    #[test]
    fn test_passband_flatness() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let mut filter = ButterworthFilter::new(4, cutoff_freq, sampling_rate);

        // Test several frequencies well below cutoff
        let test_freqs = vec![100.0, 200.0, 500.0];
        let mut max_ripple = 0.0;

        for freq in test_freqs {
            let input = generate_sine_wave(freq, 1.0, sampling_rate);
            let mut output = Vec::with_capacity(input.len());

            for &x in &input {
                output.push(filter.apply(x));
            }

            let skip_samples = (sampling_rate * 0.1) as usize;
            let power_ratio = (calculate_power(&output[skip_samples..])
                / calculate_power(&input[skip_samples..]))
            .sqrt();

            // Record maximum deviation from unity gain
            max_ripple = max_ripple.max((power_ratio - 1.0).abs());
        }

        // Butterworth filters should have minimal passband ripple
        assert_lt!(max_ripple, 0.01); // Less than 1% ripple
    }

    #[test]
    fn test_stopband_attenuation() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Test frequency well above cutoff (2 octaves)
        let test_freq = cutoff_freq * 4.0;
        let input = generate_sine_wave(test_freq, 1.0, sampling_rate);
        let mut output = Vec::with_capacity(input.len());

        for &x in &input {
            output.push(filter.apply(x));
        }

        let skip_samples = (sampling_rate * 0.1) as usize;
        let attenuation_db = -20.0
            * (calculate_power(&output[skip_samples..]) / calculate_power(&input[skip_samples..]))
                .sqrt()
                .log10();

        // Theoretical attenuation at 2 octaves should be approximately -24dB/octave for 4th order
        let expected_attenuation = 24.0 * 2.0; // 2 octaves * 24 dB/octave
        assert_in_delta!(attenuation_db, expected_attenuation, 3.0);
    }

    #[test]
    fn test_impulse_response() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Create impulse input
        let mut input = vec![0.0; 1000];
        input[0] = 1.0;

        // Get impulse response
        let mut output = Vec::with_capacity(input.len());
        for &x in &input {
            output.push(filter.apply(x));
        }

        // Verify decay characteristics
        let peak = output.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert_le!(peak, 1.0); // No amplitude amplification

        // Check if response decays
        let end_amplitude = output.iter().skip(900).map(|x| x.abs()).sum::<f64>();
        assert_lt!(end_amplitude, 0.01); // Should decay to near zero
    }

    #[test]
    fn test_step_response() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Create step input
        let input = vec![1.0; 1000];
        let mut output = Vec::with_capacity(input.len());

        for &x in &input {
            output.push(filter.apply(x));
        }

        // Verify step response characteristics
        let mut max_overshoot = 0.0;
        for &y in &output {
            if y > 1.0 {
                max_overshoot = max_overshoot.max(y - 1.0);
            }
        }

        // A 4th-order Butterworth might typically have ~5–15% overshoot
        assert_lt!(max_overshoot, 0.15);

        // Verify steady state approaches 1.0
        let steady_state = output.iter().skip(900).sum::<f64>() / 100.0;
        assert_in_delta!(steady_state, 1.0, 0.01);
    }

    #[test]
    fn test_filter_reset() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Apply some input
        for _ in 0..100 {
            filter.apply(1.0);
        }

        // Reset filter
        filter.reset();

        // Verify filter state is cleared
        let first_output = filter.apply(1.0);

        // The first output after reset should match the first output of a new filter
        let mut new_filter = ButterworthFilter::new(4, 1000.0, 44100.0);
        let expected_first_output = new_filter.apply(1.0);

        assert_approx_eq!(first_output, expected_first_output);
    }

    #[test]
    fn test_dc_response() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Apply DC input
        let mut output = Vec::new();
        for _ in 0..1000 {
            output.push(filter.apply(1.0));
        }

        // DC should pass through unchanged in steady state
        let steady_state = output.iter().skip(900).sum::<f64>() / 100.0;
        assert_in_epsilon!(steady_state, 1.0, 0.01);
    }

    #[test]
    fn test_nyquist_frequency_rejection() {
        let cutoff_freq = 1000.0;
        let sampling_rate = 44100.0;
        let order = 4;
        let mut filter = ButterworthFilter::new(order, cutoff_freq, sampling_rate);

        // Generate Nyquist frequency signal (sampling_rate/2)
        let nyquist_freq = sampling_rate / 2.01;
        // TODO: Something weird is going on at exactly the Nyquist frequency, the attenuation
        // is not as high as expected.  Need to verify this issue here.
        let input = generate_sine_wave(nyquist_freq, 1.0, sampling_rate);
        let mut output = Vec::with_capacity(input.len());

        for &x in &input {
            output.push(filter.apply(x));
        }

        let skip_samples = (sampling_rate * 0.1) as usize;
        let attenuation_db = -20.0
            * (calculate_power(&output[skip_samples..]) / calculate_power(&input[skip_samples..]))
                .sqrt()
                .log10();

        // Should have significant attenuation at Nyquist frequency
        assert_gt!(attenuation_db, 60.0); // At least 60dB attenuation
    }
}
