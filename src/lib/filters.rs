use num_traits::{Float, FloatConst, Inv};

/// An IIR digital filter implementation
pub struct IIRFilter<T: Float + FloatConst + Inv, const N: usize> {
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

impl<T: Float + FloatConst + Inv, const N: usize> IIRFilter<T, N> {
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
    T: Float + FloatConst + Inv,
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
    T: Float + FloatConst + Inv,
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
