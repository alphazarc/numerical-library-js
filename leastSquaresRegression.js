import * as tf from "@tensorflow/tfjs";

export function LSR(matrixA, shapeA, vectorB, shapeB) {
  return tf.tidy(() => {
    const A = tf.tensor(matrixA, shapeA);
    console.log("Matrix A: ");
    A.print();
    const b = tf.tensor(vectorB, shapeB);
    console.log("Vector b: ");
    b.print();
    // Least Squares Regression
    // Ax = b has unique least-square solution
    // The columns of A are linearly independent
    // A^T * A is invertible
    // x = (A^T * A)^-1 * A^T * b
    const partial_x = tf.dot(tf.transpose(A), A); // (A^T * A)

    const partial_x1 = inverse(partial_x.arraySync()); // (A^T * A)^-1

    const partial_x2 = tf.tensor(partial_x1);

    const partial_x3 = tf.dot(partial_x2, tf.transpose(A)); // (A^T * A)^-1 * A^T

    const x = tf.dot(partial_x3, b); // x = (A^T * A)^-1 * A^T * b
    console.log("x is equal to: ");
    return x.print();
  });
}

// Matrix inverse calculator
// Reference to https://gist.github.com/husa/5652439
export function inverse(A) {
  var temp,
    N = A.length,
    E = [];

  for (var i = 0; i < N; i++) E[i] = [];

  for (i = 0; i < N; i++)
    for (var j = 0; j < N; j++) {
      E[i][j] = 0;
      if (i == j) E[i][j] = 1;
    }

  for (var k = 0; k < N; k++) {
    temp = A[k][k];

    for (var j = 0; j < N; j++) {
      A[k][j] /= temp;
      E[k][j] /= temp;
    }

    for (var i = k + 1; i < N; i++) {
      temp = A[i][k];

      for (var j = 0; j < N; j++) {
        A[i][j] -= A[k][j] * temp;
        E[i][j] -= E[k][j] * temp;
      }
    }
  }

  for (var k = N - 1; k > 0; k--) {
    for (var i = k - 1; i >= 0; i--) {
      temp = A[i][k];

      for (var j = 0; j < N; j++) {
        A[i][j] -= A[k][j] * temp;
        E[i][j] -= E[k][j] * temp;
      }
    }
  }

  for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) A[i][j] = E[i][j];
  return A;
}

// inputs should be an array like [1,2,3,4]
// first input the matrix then define the shape of it
// LSR(matrixA, shapeA, vectorB, shapeB)
LSR(
  [6, 8, 4, 2, 1, 5, 12, 67, -12, 47, -77, 8],
  [6, 2],
  [11, 2, 17, 33, -6, 40],
  [6, 1]
);
