// Jacobi Method for eigen values
// Reference to https://github.com/vuoriov4/jacobi-eigenvalue-algorithm-js/blob/master/src/index.js
import * as tf from "@tensorflow/tfjs";

const eigJacobi = (input, epsilon, iterations) => {
  let n = input[0].length; // Size of the input matrix
  const matrixReshape = tf.tensor(input);
  const reshape = matrixReshape.reshape([n, n]);
  input = reshape.arraySync();
  let cloneInput = clone(input); // Clone of the input matrix as different variable
  let eye = identity(n); // Create identity matrix from the size of input matrix
  for (let i = 0; i < iterations; i++) {
    let itr = iterate(eye, cloneInput, n);
    eye = itr.eye;
    cloneInput = itr.cloneInput;
    if (iscloneInputiagonal(cloneInput, epsilon)) {
      cloneInput = clean(cloneInput, epsilon);
      eye = clean(eye, epsilon);
      break;
    }
  }
  return console.log("Eigen values are: ", [cloneInput]);
};

const iterate = (eye, cloneInput, n) => {
  // find the indices of the largest off-diagonal element (in magnitude) from cloneInput
  let di;
  let dj;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i == j) continue;
      if (
        di === undefined ||
        dj === undefined ||
        Math.abs(cloneInput[i][j]) > Math.abs(cloneInput[di][dj])
      ) {
        di = i;
        dj = j;
      }
    }
  }
  // find the rotational angle
  let angle;
  if (cloneInput[di][di] === cloneInput[dj][dj]) {
    if (cloneInput[di][dj] > 0) angle = Math.PI / 4;
    else angle = -Math.PI / 4;
  } else {
    angle =
      0.5 *
      Math.atan(
        (2 * cloneInput[di][dj]) / (cloneInput[di][di] - cloneInput[dj][dj])
      );
  }
  // compute eye1
  let eye1 = identity(n);
  eye1[di][di] = Math.cos(angle);
  eye1[dj][dj] = eye1[di][di];
  eye1[di][dj] = -Math.sin(angle);
  eye1[dj][di] = -eye1[di][dj];
  // set new values
  return {
    eye: multiply(eye, eye1),
    cloneInput: multiply(multiply(transpose(eye1), cloneInput), eye1),
  };
};

const clean = (input, epsilon) => {
  return tf.tidy(() => {
    let n = input[0].length;
    let result = clone(input);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (tf.abs(tf.tensor(input[i][j])).arraySync() < epsilon)
          result[i][j] = 0;
        else result[i][j] = input[i][j];
      }
    }
    return result;
  });
};

const iscloneInputiagonal = (input, epsilon) => {
  return tf.tidy(() => {
    let n = input[0].length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i == j) continue;
        if (tf.abs(tf.tensor(input[i][j])).arraySync() > epsilon) return false;
      }
    }
    return true;
  });
};

const multiply = (A, B) => {
  return tf.tidy(() => {
    let n = A[0].length;
    let result = identity(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          const a = tf.tensor(A[i][k]);
          const b = tf.tensor(B[k][j]);
          sum += tf.mul(a, b).arraySync();
        }
        result[i][j] = sum;
      }
    }
    return result;
  });
};
// Compute transpose of the input
const transpose = (input) => {
  return tf.tidy(() => {
    const c = tf.tensor(input);
    let cln = c.transpose().arraySync();
    return cln;
  });
};
// Create a clone matrix of the input
const clone = (input) => {
  return tf.tidy(() => {
    const b = tf.tensor(input);
    let result = b.clone().arraySync();
    return result;
  });
};
// Create identity matrix with the same size of the input matrix
const identity = (n) => {
  return tf.tidy(() => {
    let result = tf.eye(n, n).arraySync();
    return result;
  });
};
// eigJacobi(input, epsilon, iterations)
// Matrix should be symmetric
// Inputs should be array, number, number
eigJacobi(
  [
    [1, 4, 8],
    [4, 1, 9],
    [8, 9, 6],
  ],
  0.000001,
  100
);
