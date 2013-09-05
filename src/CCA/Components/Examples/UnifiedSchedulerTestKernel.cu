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
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
//no linker for device code, need to include the whole source...
#include <CCA/Components/Schedulers/GPUDataWarehouse.cu>

namespace Uintah {
//______________________________________________________________________
//
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param domainHigh a three component vector that gives the highest non-ghost layer cell of the domain as (x,y,z)
// @param domainSize a three component vector that gives the size of the domain including ghost nodes
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
__global__ void unifiedSchedulerTestKernel(int patchID,
                                           int matlIndex,
                                           uint3 domainLow,
                                           uint3 domainHigh,
                                           GPUDataWarehouse *old_gpudw,
                                           GPUDataWarehouse *new_gpudw) {
  GPUGridVariable<double> phi;
  GPUGridVariable<double> newphi;
  old_gpudw->get(phi, "phi", patchID, matlIndex);
  new_gpudw->get(newphi, "phi", patchID, matlIndex);
  // calculate the thread indices
  int i = blockDim.x * blockIdx.x + threadIdx.x + domainLow.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + domainLow.y;

  // If the threads are within the bounds of the ghost layers
  //  the algorithm is allowed to stream along the z direction
  //  applying the stencil to a line of cells.  The z direction
  //  is streamed because it allows access of x and y elements
  //  that are close to one another which should allow coalesced
  //  memory accesses.
  if(i < domainHigh.x && j < domainHigh.y) {
    for (int k = domainLow.z; k < domainHigh.z; k++) {
      
      newphi[make_int3(i,j,k)] = (1. / 6)
                  * (phi[make_int3(i-1, j, k)]
                   + phi[make_int3(i+1, j, k)]
                   + phi[make_int3(i, j-1, k)]
                   + phi[make_int3(i, j+1, k)]
                   + phi[make_int3(i, j, k-1)]
                   + phi[make_int3(i, j, k+1)]);
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
                                      GPUDataWarehouse* new_gpudw
                                      )
{
  unifiedSchedulerTestKernel<<< dimGrid, dimBlock, 0, *stream >>>(patchID,
                                                                  matlIndex,
                                                                  domainLow,
                                                                  domainHigh,
                                                                  old_gpudw,
                                                                  new_gpudw
                                                                  );
}

} //end namespace Uintah
