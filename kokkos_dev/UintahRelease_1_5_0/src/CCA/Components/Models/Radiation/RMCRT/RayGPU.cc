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
//----- RayGPU.cu ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <Core/Grid/DbgOutput.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAY_GPU", false);


//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup and invoke ray trace kernel
//---------------------------------------------------------------------------
void Ray::rayTraceGPU(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      int device,
                      bool modifies_divQ,
                      Task::WhichDW which_abskg_dw,
                      Task::WhichDW which_sigmaT4_dw,
                      Task::WhichDW which_celltype_dw)
{
  // setup for driver API kernel launch
  CUresult   cuErrVal;
  CUmodule   cuModule;
  CUfunction rayTraceKernel;

  // initialize the driver API
  CUDA_DRV_SAFE_CALL( cuErrVal = cuInit(0) )

  // set the CUDA device and context
  CUDA_RT_SAFE_CALL( cudaSetDevice(device) );

  // Single material now, but can't assume 0, need the specific ARCHES or ICE material here
  int matl = matls->getVector().front();
  int numPatches = patches->size();

  const Level* level = getLevel(patches);
  IntVector dLo, dHi;
  level->findInteriorCellIndexRange(dLo, dHi);
  uint3 domainLo = make_uint3(dLo.x(), dLo.y(), dLo.z());
  uint3 domainHi = make_uint3(dLo.x(), dHi.y(), dHi.z());

  // requires and computes on device
  double* d_absk = NULL;
  double* d_sigmaT4 = NULL;
  double* d_divQ = NULL;

  // patch loop
  for (int p = 0; p < numPatches; p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    // pointers to device-side grid-variables
    d_absk    = _scheduler->getDeviceRequiresPtr(d_abskgLabel, matl, patch);
    d_sigmaT4 = _scheduler->getDeviceRequiresPtr(d_sigmaT4_label, matl, patch);
    d_divQ    = _scheduler->getDeviceComputesPtr(d_divQLabel, matl, patch);

    // Calculate the memory block size
    IntVector nec = patch->getExtraCells();
    IntVector l = patch->getCellLowIndex();
    IntVector h = patch->getCellHighIndex();

    IntVector divQSize = _scheduler->getDeviceComputesSize(d_divQLabel, matl, patch);
    int xdim = divQSize.x();
    int ydim = divQSize.y();
    int zdim = divQSize.z();

    // get the cell spacing and convert to CUDA vector type
    Vector dcell = patch->dCell();
    const double3 cellSpacing = make_double3(dcell.x(), dcell.y(), dcell.z());

    // Patch extents used by the kernel to prevent out of bounds accesses.
    const uint3 patchLo = make_uint3(l.x(), l.y(), l.z());
    const uint3 patchHi = make_uint3(h.x(), h.y(), h.z());
    const uint3 patchSize = make_uint3(xdim, ydim, zdim);

    // Set up number of thread blocks in X and Y directions accounting for dimensions not divisible by 8
    int xBlocks = ((xdim % 8) == 0) ? (xdim / 8) : ((xdim / 8) + 1);
    int yBlocks = ((ydim % 8) == 0) ? (ydim / 8) : ((ydim / 8) + 1);
    dim3 dimGrid(xBlocks, yBlocks, 1); // grid dimensions (blocks per grid))

    // block dimensions (threads per block)
    int tpbX = 8;
    int tpbY = 8;
    int tpbZ = 1;
    dim3 dimBlock(tpbX, tpbY, tpbZ);

    // setup random number generator states on the device, 1 for each thread
    curandState* globalDevStates;
    int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
    CUDA_RT_SAFE_CALL( cudaMalloc((void**)&globalDevStates, numStates * sizeof(curandState)) );

    // setup and launch kernel
    void *kernelParms[] = { (void*)(&patchLo), (void*)(&patchHi), (void*)(&patchSize), (void*)(&domainLo), (void*)(&domainHi),
                            (void*)(&cellSpacing), &d_absk, &d_sigmaT4, &d_divQ, &_virtRad, &_isSeedRandom, &_CCRays,
                            &_NoOfRays, &_viewAng, &_Threshold, &globalDevStates };
    string ptxpath = string(PTX_DIR_PATH)+"/RayGPUKernel.ptx";
    CUDA_DRV_SAFE_CALL( cuErrVal = cuModuleLoad(&cuModule, ptxpath.c_str()) );
    CUDA_DRV_SAFE_CALL( cuErrVal = cuModuleGetFunction(&rayTraceKernel, cuModule, "rayTraceKernel") );
    cudaStream_t* stream = _scheduler->getCudaStream(device);

    // launch the kernel
    cuErrVal = cuLaunchKernel(rayTraceKernel, dimGrid.x, dimGrid.y, dimGrid.z,
                              dimBlock.x, dimBlock.y, dimBlock.z, 0, *stream, kernelParms, 0);

    // get updated divQ back into host memory
    cudaEvent_t* event = _scheduler->getCudaEvent(device);
    _scheduler->requestD2HCopy(d_divQLabel, matl, patch, stream, event);

    // free device-side RNG states
    CUDA_RT_SAFE_CALL( cudaFree(globalDevStates) );

  }  //end patch loop
}  // end GPU ray trace method
