import * as tf from "@tensorflow/tfjs";
// Derivative Calculator for analytical derivatives using numerical differentiation
export function numDiff(h, v, t) {
  return tf.tidy(() => {
    // Array creator with desired boundries
    const range = (start, stop, step) =>
      Array.from(
        { length: (stop - start) / step + 1 },
        (_, i) => start + i * step
      );
    // Define the grid
    const x = range(v, t, h);
    // Compute the function.
    const y = x.map(Math.cos);
    // Function to calculate difference between each element in array
    function diff(A) {
      return A.slice(1).map(function (n, m) {
        return n - A[m];
      });
    }
    // compute forward difference
    const forward_diff = diff(y).map((k) => k / h);

    const x_diff = x.slice(0, -1);
    // compute the exact solution
    const exact_solution = x_diff.map(Math.sin).map((j) => j * -1);
    const forward_tf = tf.tensor(forward_diff);
    const exact_tf = tf.tensor(exact_solution);
    const max_error = tf.max(tf.abs(tf.sub(exact_tf, forward_tf)));
    console.log("Forward difference is: ", forward_diff);
    return console.log("max error: " + max_error.arraySync());
  });
}
// Computes the forward differences
// numDiff(h, v, t)
// h is the step size for our grid
// v is the starting point of the grid
// t is the ending point of the grid
numDiff(0.1, 0, 2 * Math.PI);
