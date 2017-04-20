/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <sci_defs/cuda_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

#define indicesSize 24
#define N_      3
#define N_xi    7
#define blocks  1024
#define threads 72

//______________________________________________________________________
//
// @brief
// @param
//
__global__ void constructLinearSystemKernel(double* weightsArray,
		                                        double* weightedAbscissasArray,
		                                        double* modelsArray,
		                                        int*    momentIndicesArray,
		                                        double* AAArray2,
		                                        double* BBArray2,
		                                        int     num_cells)
{

  int bx, tx;
  bx = blockIdx.x;
  tx = threadIdx.x;
  int gd, bd;
  gd = gridDim.x;
  bd = blockDim.x;
  int ki, nk, mi, nm;
  nm = 3;
  mi = tx % nm;
  nk = 24;
  ki = tx / nm % nk;

  //preprocessing
  __shared__ double d_powers[N_xi][indicesSize * N_];
  __shared__ double powers[N_xi][indicesSize * N_];
  __shared__ double totalsumS[indicesSize * N_];
  //__shared__ float rightPartialProduct[N_xi][indicesSize*N_], leftPartialProduct[N_xi][indicesSize*N_];

  unsigned int cell_num, iter, k, n, m, j, alpha, i;

  // construct AX=B
  for (cell_num = bx; cell_num < num_cells; cell_num += gd) {

    double* weights = &(weightsArray[cell_num * N_]);
    double* weightedAbscissas = &(weightedAbscissasArray[cell_num * N_ * N_xi]);
    double* models = &(modelsArray[cell_num * N_ * N_xi]);
    double* AA = &(AAArray2[cell_num * indicesSize * N_ * N_xi]);
    double* BB = &(BBArray2[cell_num * indicesSize]);

    // construct AX=B
    for (k = ki; k < indicesSize; k += nk) {

      int* thisMoment = &(momentIndicesArray[k * N_xi]);

      for (m = mi; m < N_; m += nm) {
        if (weights[m] != 0) {
          for (n = 0; n < N_xi; n++) {
            double base = weightedAbscissas[n * N_ + m] / weights[m];
            double exponent = thisMoment[n] - 1;
            double power = 1;
            for (i = 0; i < exponent; i++) {
              power *= base;
            }
            d_powers[n][k * N_ + m] = power;
            powers[n][k * N_ + m] = power * base;
            if (exponent == -1) {
              powers[n][k * N_ + m] = 1;
            }
          }
        } else {
          for (n = 0; n < N_xi; n++) {
            d_powers[n][k * N_ + m] = powers[n][k * N_ + m] = 0;
          }
        }
      }
      __syncthreads();

//      // partial products
//      for (m = mi; m < N_; m += nm) {
//        rightPartialProduct[0][k * N_ + m] = 1;
//        leftPartialProduct[N_xi - 1][k * N_ + m] = 1;
//        for (n = 1; n < N_xi; n++) {
//          rightPartialProduct[n][k * N_ + m] = rightPartialProduct[n - 1][k * N_ + m] * powers[n - 1][k * N_ + m];
//          leftPartialProduct[N_xi - 1 - n][k * N_ + m] = leftPartialProduct[N_xi - 1 - n + 1][k * N_ + m]
//              * powers[N_xi - 1 - n + 1][k * N_ + m];
//        }
//      }

      //no __syncthreads() needed here as last one covers race conditions

      // weights
      for (alpha = mi; alpha < N_; alpha += nm) {
        double prefixA = 1;
        double productA = 1;
        for (i = 0; i < N_xi; ++i) {
          // Appendix C, C.9 (A1 matrix)
          prefixA = prefixA - (thisMoment[i]);
          productA = productA * powers[i][k * N_ + alpha];
        }
        AA[k * N_ * N_xi + alpha] = prefixA * productA;
      } //end weights sub-matrix

      // weighted abscissas
      //__syncthreads(); //for partial products
      double localTotalsumS = 0;
      for (alpha = mi; alpha < N_; alpha += nm) {

        double prefixA = 1;
        double productA = 1;

        double prefixS = 1;
        double productS = 1;
        double modelsumS = 0;

        double quadsumS = 0;
        for (j = 0; j < N_xi; ++j) {
          prefixA = (thisMoment[j]) * d_powers[j][k * N_ + alpha];

          // Appendix C, C.16 (S matrix)
          prefixS = -(thisMoment[j]) * d_powers[j][k * N_ + alpha];

          productA = productS = 1;

          for (n = 0; n < N_xi; ++n) {
            if (n != j) {
              if (weights[alpha] == 0) {
                productS = productA = 0;
              } else {
                productA = productA * (powers[n][k * N_ + alpha]);
                productS = productS * (powers[n][k * N_ + alpha]);
              }
            }
          }

          //productS=productA=rightPartialProduct[j][k*N_+alpha]*leftPartialProduct[j][k*N_+alpha];

          modelsumS = -models[j * (N_) + alpha];

          int col = (j + 1) * N_ + alpha;
          if (j + 1 < N_xi)
            AA[k * N_ * N_xi + col] = prefixA * productA;

          quadsumS = quadsumS + weights[alpha] * modelsumS * prefixS * productS;
        } //end quad nodes

        localTotalsumS = localTotalsumS + quadsumS;

      } //end int coords j sub-matrix

      totalsumS[tx] = localTotalsumS;

      __syncthreads();

      if (mi == 0) {
        localTotalsumS = totalsumS[tx] + totalsumS[tx + 1] + totalsumS[tx + 2];
        BB[k] = localTotalsumS;
      }

    } // end moments
  } // end cells
}

//______________________________________________________________________
//
// @brief
// @param
//
void launchConstructLinearSystemKernel(double* weightsArray,
                                       double* weightedAbscissasArray,
                                       double* modelsArray,
                                       int*    momentIndicesArray,
                                       double* AAArray,
                                       double* BBArray,
                                       int     num_cells)
{
  double* d_weights;
  double* d_weightedAbscissas;
  double* d_models;
  double* d_AA;
  double* d_BB;
  int*    d_momentIndices;

  // allocate memory on the device for arguments
  cudaMalloc(&d_weights, N_ * num_cells * sizeof(double));
  cudaMalloc(&d_weightedAbscissas, N_xi * N_ * num_cells * sizeof(double));
  cudaMalloc(&d_models, N_xi * N_ * num_cells * sizeof(double));
  cudaMalloc(&d_momentIndices, indicesSize * N_xi * sizeof(int));
  cudaMalloc(&d_AA, indicesSize * N_xi * N_ * num_cells * sizeof(double));
  cudaMalloc(&d_BB, indicesSize * num_cells * sizeof(double));

  // host-to-device memcopies
  cudaMemcpy(d_weights, weightsArray, N_ * num_cells * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weightedAbscissas, weightedAbscissasArray, N_xi * N_ * num_cells * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_models, modelsArray, N_xi * N_ * num_cells * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_momentIndices, momentIndicesArray, indicesSize * N_xi * sizeof(int), cudaMemcpyHostToDevice);

  double* zeros = (double*) malloc(num_cells * indicesSize * N_ * N_xi * sizeof(double));
  memset((void*) zeros, 0, num_cells * indicesSize * N_ * N_xi * sizeof(double));
  cudaMemcpy(d_AA, zeros, num_cells * indicesSize * N_ * N_xi * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_BB, zeros, num_cells * indicesSize * sizeof(double), cudaMemcpyHostToDevice);
  free(zeros);

  // launch the system construction kernel
  constructLinearSystemKernel<<<blocks, threads>>>(d_weights,
                                                   d_weightedAbscissas,
                                                   d_models,
                                                   d_momentIndices,
                                                   d_AA,
                                                   d_BB,
                                                   num_cells);

  // block until all work on the device is complete
  cudaThreadSynchronize();

  // copy results back to host
  cudaMemcpy(AAArray, d_AA, num_cells * indicesSize * N_ * N_xi * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(BBArray, d_BB, num_cells * indicesSize * sizeof(double), cudaMemcpyDeviceToHost);

  // cleanup allocated memory
  cudaFree(d_weights);
  cudaFree(d_weightedAbscissas);
  cudaFree(d_models);
  cudaFree(d_momentIndices);
  cudaFree(d_AA);
  cudaFree(d_BB);
}

#ifdef __cplusplus
}
#endif

