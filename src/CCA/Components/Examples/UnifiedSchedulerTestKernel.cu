/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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
#include <Core/Geometry/GPUVector.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
// linker support for device code not ready yet, need to include the whole source...
#include <CCA/Components/Schedulers/GPUDataWarehouse.cu>

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
__global__ void unifiedSchedulerTestKernel(int patchID,
                                           int matlIndex,
                                           uint3 domainLow,
                                           uint3 domainHigh,
                                           GPUDataWarehouse* old_gpudw,
                                           GPUDataWarehouse* new_gpudw) {

  const GPUGridVariable<double> phi;
  GPUGridVariable<double> newphi;
  old_gpudw->get(phi, "phi", patchID, matlIndex);
  new_gpudw->getModifiable(newphi, "phi", patchID, matlIndex);

  // calculate the thread indices
  int i = blockDim.x * blockIdx.x + threadIdx.x + domainLow.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + domainLow.y;

  // If the threads are within the bounds of the patch
  //  the algorithm is allowed to stream along the z direction
  //  applying the stencil to a line of cells.  The z direction
  //  is streamed because it allows access of x and y elements
  //  that are close to one another which should allow coalesced
  //  memory accesses.
  if(i < domainHigh.x && j < domainHigh.y) {
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

void launchUnifiedSchedulerTestKernel(dim3 dimGrid,
                                      dim3 dimBlock,
                                      cudaStream_t* stream,
                                      int patchID,
                                      int matlIndex,
                                      uint3 domainLow,
                                      uint3 domainHigh,
                                      GPUDataWarehouse* old_gpudw,
                                      GPUDataWarehouse* new_gpudw)
{
  unifiedSchedulerTestKernel<<< dimGrid, dimBlock, 0, *stream >>>(patchID,
                                                                  matlIndex,
                                                                  domainLow,
                                                                  domainHigh,
                                                                  old_gpudw,
                                                                  new_gpudw);
}

} //end namespace Uintah
