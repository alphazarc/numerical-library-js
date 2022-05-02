import * as tf from "@tensorflow/tfjs";
// Wrapper function for Fast Fourier Transformation with tensorflow.js
export function FFT(real, imag) {
  return tf.tidy(() => {
    const x = tf.complex(real, imag);

    return x.fft().print();
  });
}
// First input represents the real part and
// Second input represents the imaginary part
// Inputs should be array
// FFT(real, imag)
FFT([1, 2, 3], [1, 2, 3]);
