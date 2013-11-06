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
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#ifdef HAVE_CUDA
#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#endif
#include <Core/Grid/DbgOutput.h>


#define BLOCKSIZE 16

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAYGPU", false);

//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup for ray trace kernel
//---------------------------------------------------------------------------
void Ray::rayTraceGPU(Task::CallBackEvent event,
                      const ProcessorGroup* pg,
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
  if (event==Task::GPU) {
#ifdef HAVE_CUDA
  cout << " top RayTraceGPU: " << endl;
  
  const Level* level = getLevel(patches);
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  if ( doCarryForward( timestep, radCalc_freq) ) {
    return;
  }

  //__________________________________
  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;
  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

  const int3 dev_domainLo = make_int3(domainLo_EC.x(), domainLo_EC.y(), domainLo_EC.z());
  const int3 dev_domainHi = make_int3(domainHi_EC.x(), domainHi_EC.y(), domainHi_EC.z());

  
  //__________________________________
  //  
  GPUDataWarehouse* old_gdw = old_dw->getGPUDW()->getdevice_ptr();
  GPUDataWarehouse* new_gdw = new_dw->getGPUDW()->getdevice_ptr();
  
  GPUDataWarehouse* abskg_gdw    = new_dw->getOtherDataWarehouse(which_abskg_dw)->getGPUDW()->getdevice_ptr();
  GPUDataWarehouse* sigmaT4_gdw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw)->getGPUDW()->getdevice_ptr();
  GPUDataWarehouse* celltype_gdw = new_dw->getOtherDataWarehouse(which_celltype_dw)->getGPUDW()->getdevice_ptr();
  
  //__________________________________
  //  varLabel name struct
  varLabelNames labelNames;
#if 0
  labelNames->abskg     = d_abskgLabel->getName().c_str();    // cuda doesn't support C++ strings
  labelNames.sigmaT4   = d_sigmaT4_label->getName().c_str();
  labelNames.divQ      = d_divQLabel->getName().c_str();
  labelNames.celltype  = d_cellTypeLabel->getName().c_str();
  labelNames.VRFlux    = d_VRFluxLabel->getName().c_str();
  labelNames.boundFlux = d_boundFluxLabel->getName().c_str();
  labelNames.radVolQ   = d_radiationVolqLabel->getName().c_str();
  
  cout << " abskg:   " << d_abskgLabel->getName() << endl;
  cout << " sigmaT4: " << d_sigmaT4_label->getName() << endl;
  cout << " divQ:    " <<d_divQLabel->getName() << endl;
  cout << " cellType:" <<d_cellTypeLabel->getName() << endl;
  cout << " VRFlux:  " << d_VRFluxLabel->getName() << endl;
  cout << " boundFlux: " << d_boundFluxLabel->getName() << endl;
  cout << " radVolQ:   " << d_radiationVolqLabel->getName() << endl;
#endif
  
  //__________________________________
  //  RMCRT_flags
  RMCRT_flags RT_flags;
  RT_flags.modifies_divQ = modifies_divQ;
  
  RT_flags.virtRad            = _virtRad;
  RT_flags.solveDivQ          = _solveDivQ;
  RT_flags.allowReflect       = _allowReflect;
  RT_flags.solveBoundaryFlux  = _solveBoundaryFlux;
  RT_flags.isSeedRandom       = _isSeedRandom;
  RT_flags.CCRays             = _CCRays;
  
  RT_flags.sigma      = _sigma;;    
  RT_flags.sigmaScat  = _sigmaScat; 
  RT_flags.threshold  = _Threshold;
  
  RT_flags.nDivQRays = _nDivQRays;   
  RT_flags.nRadRays  = _nRadRays;    
  RT_flags.nFluxRays = _nFluxRays;
  
  double start=clock();  
  
  //______________________________________________________________________
  //
  // patch loop
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    // Calculate the memory block size
    const IntVector low = patch->getCellLowIndex();
    const IntVector high = patch->getCellHighIndex();
    const IntVector patchSize = high - low;

    const int xdim = patchSize.x();
    const int ydim = patchSize.y();
    const int zdim = patchSize.z();

    // get the cell spacing and convert patch extents to CUDA vector type
    patchParams patchP;
    const Vector dx = patch->dCell();
    patchP.dx     = make_double3(dx.x(), dx.y(), dx.z());
    patchP.lo     = make_int3(low.x(), low.y(), low.z());
    patchP.hi     = make_int3(high.x(), high.y(), high.z());
    patchP.ID     = patch->getID();
    patchP.nCells = make_int3(xdim, ydim, zdim);

    // define dimesions of the thread grid to be launched
    int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
    int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(xblocks, yblocks, 1);

    // setup random number generator states on the device, 1 for each thread
    curandState* randNumStates;
    int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
    /*`CUDA_RT_SAFE_CALL( cudaMalloc((void**)&randNumStates, numStates * sizeof(curandState)) );      TESTING`*/

    
    RT_flags.nRaySteps = 0;

    //__________________________________
    // set up and launch kernel
    launchRayTraceKernel(dimGrid, 
                         dimBlock,
                         d_matl,
                         patchP,
                         dev_domainLo, 
                         dev_domainHi, 
                         randNumStates, 
                         (cudaStream_t*)stream,
                         RT_flags,
                         labelNames,
                         abskg_gdw, 
                         sigmaT4_gdw, 
                         celltype_gdw, 
                         old_gdw, 
                         new_gdw);

    // free device-side RNG states
    /*`CUDA_RT_SAFE_CALL( cudaFree(randNumStates) );      TESTING`*/
    
    //__________________________________
    //
    double end =clock();
    double efficiency = RT_flags.nRaySteps/((end-start)/ CLOCKS_PER_SEC);
    
    if (patch->getGridIndex() == 0) {
      cout<< endl;
      cout << " RMCRT REPORT: Patch 0" << endl;
      cout << " Used "<< (end-start) * 1000 / CLOCKS_PER_SEC<< " milliseconds of CPU time. \n" << endl;// Convert time to ms
      cout << " Size: " << RT_flags.nRaySteps << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
  }  //end patch loop
#endif
  } //end GPU task code


}  // end GPU ray trace method
