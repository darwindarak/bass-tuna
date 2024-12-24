use num_traits::{Float, FloatConst};

/// A single-pole IIR lowpass filter.
pub struct Lowpass<T: Float + FloatConst> {
    alpha: T,

    y_1: T, // internal state to store the previous output
}

impl<T: Float + FloatConst> Lowpass<T> {
    /// Creates a new lowpass filter.
    ///
    /// # Arguments
    /// * `sample_rate` - The sampling rate of the input signal in Hz.
    /// * `cutoff_frequency` - The cutoff frequency for the lowpass filter in Hz.
    pub fn new(sample_rate: usize, cutoff_frequency: T) -> Self {
        let omega =
            T::from(2).unwrap() * T::PI() * cutoff_frequency / T::from(sample_rate).unwrap();
        let alpha = omega / (omega + T::one());

        Lowpass {
            alpha,
            y_1: T::zero(),
        }
    }

    /// Applies the lowpass filter to a single sample.
    ///
    /// # Arguments
    /// * `x` - The current input sample.
    ///
    /// # Returns
    /// The filtered output.
    pub fn apply(&mut self, x: T) -> T {
        let y = self.y_1 + self.alpha * (x - self.y_1);
        self.y_1 = y;

        y
    }

    pub fn latest_output(&self) -> T {
        self.y_1
    }
}

/// A biquad filter implementation for resonator filtering.
pub struct BiquadFilter<T: Float + FloatConst> {
    b_0: T, // Feedforward coefficient
    b_1: T,
    b_2: T,
    a_1: T, // Feedback coefficient
    a_2: T,

    y_1: T, // Previous output sample (y[n-1])
    y_2: T, // Output sample (y[n-2])
    x_1: T, // Previous input sample (x[n-1])
    x_2: T, // Input sample (x[n-2])
}

impl<T: Float + FloatConst> BiquadFilter<T> {
    /// Applies the filter to a single input sample and returns the output sample.
    ///
    /// # Arguments
    /// * `x` - The input sample.
    ///
    /// # Returns
    /// The filtered output sample.
    pub fn apply(&mut self, x: T) -> T {
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
    pub fn new(f0: T, sample_rate: usize, q: T) -> Self {
        let k = f0 / T::from(sample_rate).unwrap();
        let omega_0 = T::from(2).unwrap() * T::PI() * k;
        let alpha = omega_0.sin() / (T::from(2).unwrap() * q);

        let a_0 = T::one() + alpha;
        let a_1 = -T::from(2).unwrap() * omega_0.cos() / a_0;
        let a_2 = (T::one() - alpha) / a_0;

        let b_0 = alpha / a_0;
        let b_1 = T::zero();
        let b_2 = -b_0;

        Self {
            b_0,
            b_1,
            b_2,
            a_1,
            a_2,
            y_1: T::zero(),
            y_2: T::zero(),
            x_1: T::zero(),
            x_2: T::zero(),
        }
    }

    pub fn latest_output(&self) -> T {
        self.y_1
    }
}
