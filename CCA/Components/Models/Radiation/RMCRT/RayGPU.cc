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
//----- RayGPU.cu ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <Core/Grid/DbgOutput.h>

#include <sci_defs/cuda_defs.h>

#define BLOCKSIZE 16

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAYGPU", false);

//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup for ray trace kernel
//---------------------------------------------------------------------------
void Ray::rayTraceGPU(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      void* stream,
                      bool modifies_divQ,
                      Task::WhichDW which_abskg_dw,
                      Task::WhichDW which_sigmaT4_dw,
                      Task::WhichDW which_celltype_dw,
                      const int radCalc_freq)
{
  // bail if not doing a radiation step
  // equivalent to CPU rayTrace conditional: if ( doCarryForward( timestep, radCalc_freq) )
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( (timestep%radCalc_freq != 0) && (timestep != 1) ) {
    return;
  }

  const Level* level = getLevel(patches);

  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;
  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

  const uint3 dev_domainLo = make_uint3(domainLo_EC.x(), domainLo_EC.y(), domainLo_EC.z());
  const uint3 dev_domainHi = make_uint3(domainHi_EC.x(), domainHi_EC.y(), domainHi_EC.z());

  // patch loop
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    // Calculate the memory block size
    const IntVector nec = patch->getExtraCells();
    const IntVector low = patch->getCellLowIndex();
    const IntVector high = patch->getCellHighIndex();
    const IntVector size = high - low;

    const int xdim = size.x();
    const int ydim = size.y();
    const int zdim = size.z();

    // get the cell spacing and convert patch extents to CUDA vector type
    const Vector dx = patch->dCell();
    const double3 cellSpacing = make_double3(dx.x(), dx.y(), dx.z());
    const uint3 dev_patchLo = make_uint3(low.x(), low.y(), low.z());
    const uint3 dev_patchHi = make_uint3(high.x(), high.y(), high.z());
    const uint3 dev_patchSize = make_uint3(xdim, ydim, zdim);

    // define dimesions of the thread grid to be launched
    int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
    int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(xblocks, yblocks, 1);

    // setup random number generator states on the device, 1 for each thread
    curandState* globalDevRandStates;
    int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
    CUDA_RT_SAFE_CALL( cudaMalloc((void**)&globalDevRandStates, numStates * sizeof(curandState)) );


    // set up and launch kernel

    launchRayTraceKernel(dimGrid, dimBlock,
                         patch->getID(), d_matl,
                         dev_patchLo, dev_patchHi, dev_patchSize,
                         dev_domainLo, dev_domainHi, cellSpacing,
                         globalDevRandStates, (cudaStream_t*)stream,
                         _virtRad, _isSeedRandom, _CCRays, _nDivQRays, _viewAng, _Threshold,
                         old_dw->getGPUDW()->getdevice_ptr(), new_dw->getGPUDW()->getdevice_ptr());

    // free device-side RNG states
    CUDA_RT_SAFE_CALL( cudaFree(globalDevRandStates) );

  }  //end patch loop
}  // end GPU ray trace method
