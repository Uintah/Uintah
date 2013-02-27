/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//______________________________________________________________________
//
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param domainHigh a three component vector that gives the highest non-ghost layer cell of the domain as (x,y,z)
// @param domainSize a three component vector that gives the size of the domain including ghost nodes
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
__global__ void gpuSchedulerTestKernel(uint3 domainLow,
                                       uint3 domainHigh,
                                       uint3 domainSize,
                                       int numGhostCells,
                                       double *phi,
                                       double *newphi)
{
  // calculate the thread indices
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  // Get the size of the data block in which the variables reside.
  //  This is essentially the stride in the index calculations.
  int dx = domainSize.x;
  int dy = domainSize.y;

  // If the threads are within the bounds of the ghost layers
  //  the algorithm is allowed to stream along the z direction
  //  applying the stencil to a line of cells.  The z direction
  //  is streamed because it allows access of x and y elements
  //  that are close to one another which should allow coalesced
  //  memory accesses.
  if(i > 0 && j > 0 && i < domainHigh.x && j < domainHigh.y) {
    for (int k = domainLow.z; k < domainHigh.z; k++) {
      // For an array of [ A ][ B ][ C ], we can index it thus:
      // (a * B * C) + (b * C) + (c * 1)
      int idx = INDEX3D(dx,dy,i,j,k);

      newphi[idx] = (1. / 6)
                  * (phi[INDEX3D(dx,dy, (i-1), j, k)]
                   + phi[INDEX3D(dx,dy, (i+1), j, k)]
                   + phi[INDEX3D(dx,dy, i, (j-1), k)]
                   + phi[INDEX3D(dx,dy, i, (j+1), k)]
                   + phi[INDEX3D(dx,dy, i, j, (k-1))]
                   + phi[INDEX3D(dx,dy, i, j, (k+1))]);
    }
  }
}

void launchGPUSchedulerTestKernel(dim3 dimGrid,
                                  dim3 dimBlock,
                                  cudaStream_t* stream,
                                  uint3 domainLow,
                                  uint3 domainHigh,
                                  uint3 domainSize,
                                  int numGhostCells,
                                  double* d_phi,
                                  double* d_newphi)
{
  gpuSchedulerTestKernel<<< dimGrid, dimBlock, 0, *stream >>>(domainLow,
                                                              domainHigh,
                                                              domainSize,
                                                              numGhostCells,
                                                              d_phi,
                                                              d_newphi);
}

#ifdef __cplusplus
}
#endif
