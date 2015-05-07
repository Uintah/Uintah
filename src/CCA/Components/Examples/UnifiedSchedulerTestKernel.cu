/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/GPU.h>

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

namespace Uintah {

//______________________________________________________________________
//
// @brief A GPU kernel for the Jacobi iterations in the Poisson 1-material solver
// @param patchID the patch this kernel will operate over
// @param matlIndex the material associated with the specified patchID
// @param domainLow a three component vector that gives the lower corner of the work area as (x,y,z)
// @param domainHigh a three component vector that gives the highest corner of the work area as (x,y,z)
// @param old_gpudw the old GPU DataWarehouse
// @param new_gpudw the new GPU DataWarehouse

__global__
void
unifiedSchedulerTestKernel( int                patchID,
                            uint3              patchNodeLowIndex,
                            uint3              patchNodeHighIndex,
                            uint3              domainLow,
                            uint3              domainHigh,
                            GPUDataWarehouse * old_gpudw,
                            GPUDataWarehouse * new_gpudw,
                            cudaStream_t     * stream)
{
  const GPUGridVariable<double> phi;
  GPUGridVariable<double> newphi;
  old_gpudw->get(phi, "phi", patchID, 0);
  new_gpudw->getModifiable(newphi, "phi", patchID, 0);

  // calculate the thread indices
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  // If the threads are within the bounds of the patch
  // the algorithm is allowed to stream along the z direction
  // applying the stencil to a line of cells.  The z direction
  // is streamed because it allows access of x and y elements
  // that are close to one another which should allow coalesced
  // memory accesses.

  // We also need to copy the boundary cells on the z faces.
  // These outer cells don't get computed, just preserved across iterations
  // newphi(i,j,k) = phi(i,j,k)

  if ((domainLow.x - patchNodeLowIndex.x == 1 && i == patchNodeLowIndex.x) ||
      (domainLow.y - patchNodeLowIndex.y == 1 && j == patchNodeLowIndex.y) ||
      (patchNodeHighIndex.x - domainHigh.x == 1 && i == patchNodeHighIndex.x - 1) ||
      (patchNodeHighIndex.y - domainHigh.y == 1 && j == patchNodeHighIndex.y - 1)) {
      for (int k = domainLow.z; k < domainHigh.z; k++) {
        newphi(i,j,k) = phi(i,j,k);
      }

  }
  if(i >= patchNodeLowIndex.x && j >= patchNodeLowIndex.y && i < patchNodeHighIndex.x && j < patchNodeHighIndex.y ) {
    if (domainLow.z - patchNodeLowIndex.z == 1){

      newphi(i,j,patchNodeLowIndex.z) = phi(i,j,patchNodeLowIndex.z);
    }
    if (patchNodeHighIndex.z - domainHigh.z == 1) {
      newphi(i,j,patchNodeHighIndex.z - 1) = phi(i,j,patchNodeHighIndex.z - 1);
    }
  }

  if(i >= domainLow.x && j >= domainLow.y && i < domainHigh.x && j < domainHigh.y ) {

    for (int k = domainLow.z; k < domainHigh.z; k++) {

      newphi(i,j,k) = (1. / 6)
                  * (phi(i-1, j, k)
                   + phi(i+1, j, k)
                   + phi(i, j-1, k)
                   + phi(i, j+1, k)
                   + phi(i, j, k-1)
                   + phi(i, j, k+1));
    }
  }
}

void
launchUnifiedSchedulerTestKernel( dim3               dimGrid,
                                  dim3               dimBlock,
                                  cudaStream_t     * stream,
                                  int                patchID,
                                  uint3              patchNodeLowIndex,
                                  uint3              patchNodeHighIndex,
                                  uint3              domainLow,
                                  uint3              domainHigh,
                                  GPUDataWarehouse * old_gpudw,
                                  GPUDataWarehouse * new_gpudw)
{
  unifiedSchedulerTestKernel<<< dimGrid, dimBlock, 0, *stream>>>( patchID,
                                                                  patchNodeLowIndex,
                                                                  patchNodeHighIndex,
                                                                  domainLow,
                                                                  domainHigh,
                                                                  old_gpudw,
                                                                  new_gpudw,
                                                                  stream );
}

} //end namespace Uintah
