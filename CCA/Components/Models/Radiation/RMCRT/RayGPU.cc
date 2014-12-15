/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
#include <Core/Grid/DbgOutput.h>
#ifdef HAVE_CUDA
  #include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#endif


#define BLOCKSIZE 16
//#define PRINTF            // if using printf statements to debug


using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAYGPU", false);

//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup for ray trace kernel
//---------------------------------------------------------------------------
template< class T >
void Ray::rayTraceGPU(Task::CallBackEvent event,
                      const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      void* stream,
                      int deviceID,
                      bool modifies_divQ,
                      Task::WhichDW which_abskg_dw,
                      Task::WhichDW which_sigmaT4_dw,
                      Task::WhichDW which_celltype_dw,
                      const int radCalc_freq)
{
  if (event==Task::GPU) {
#ifdef HAVE_CUDA
  
  if ( doCarryForward( radCalc_freq) ) {
    return;
  }

  //__________________________________
  //  increase the size of the printbuffer on the device
#ifdef PRINTF
  size_t size;
  cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*size );
  printf(" Increasing the size of the print buffer to %d bytes\n", (int)10 * size );
#endif
  
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
  labelNames->abskg    = d_abskgLabel->getName().c_str();    // cuda doesn't support C++ strings
  labelNames.sigmaT4   = d_sigmaT4_label->getName().c_str();
  labelNames.divQ      = d_divQLabel->getName().c_str();
  labelNames.celltype  = d_cellTypeLabel->getName().c_str();
  labelNames.VRFlux    = d_VRFluxLabel->getName().c_str();
  labelNames.boundFlux = d_boundFluxLabel->getName().c_str();
  labelNames.radVolQ   = d_radiationVolqLabel->getName().c_str();
  
  cout << " abskg:   " << d_abskgLabel->getName() << endl;
  cout << " sigmaT4: " << d_sigmaT4_label->getName() << endl;
  cout << " divQ:    " << d_divQLabel->getName() << endl;
  cout << " cellType:" << d_cellTypeLabel->getName() << endl;
  cout << " VRFlux:  " << d_VRFluxLabel->getName() << endl;
  cout << " boundFlux: " << d_boundFluxLabel->getName() << endl;
  cout << " radVolQ:   " << d_radiationVolqLabel->getName() << endl;
#endif
  
  //__________________________________
  //  RMCRT_flags
  RMCRT_flags RT_flags;
  RT_flags.modifies_divQ = modifies_divQ;
  
  RT_flags.solveDivQ          = d_solveDivQ;
  RT_flags.allowReflect       = d_allowReflect;
  RT_flags.solveBoundaryFlux  = d_solveBoundaryFlux;
  RT_flags.CCRays             = d_CCRays;
  
  RT_flags.sigma      = d_sigma;    
  RT_flags.sigmaScat  = d_sigmaScat; 
  RT_flags.threshold  = d_threshold;
  
  RT_flags.nDivQRays = d_nDivQRays;
  RT_flags.nFluxRays = d_nFluxRays;
  
  double start=clock();  
  
  //______________________________________________________________________
  //
  // patch loop
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; ++p) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    // Calculate the memory block size
    const IntVector loEC = patch->getExtraCellLowIndex();
    const IntVector lo   = patch->getCellLowIndex();
    
    const IntVector hiEC = patch->getExtraCellHighIndex();
    const IntVector hi   = patch->getCellHighIndex();
    const IntVector patchSize = hiEC - loEC;

    const int xdim = patchSize.x();
    const int ydim = patchSize.y();
    const int zdim = patchSize.z();

    // get the cell spacing and convert patch extents to CUDA vector type
    patchParams patchP;
    const Vector dx = patch->dCell();
    patchP.dx     = make_double3(dx.x(), dx.y(), dx.z());
    patchP.lo     = make_int3( lo.x(), lo.y(), lo.z() );
    patchP.hi     = make_int3( hi.x(), hi.y(), hi.z() );
    
    patchP.loEC   = make_int3( loEC.x(), loEC.y(),  loEC.z() );
    patchP.hiEC   = make_int3( hiEC.x(), hiEC.y(),  hiEC.z() );
    
    patchP.ID     = patch->getID();
    patchP.nCells = make_int3(xdim, ydim, zdim);
    
    // define dimensions of the thread grid to be launched
    int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
    int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
    
    // if the # cells in a block < BLOCKSIZE^2 reduce block size
    int blocksize = BLOCKSIZE;
    if( xblocks == 1 && yblocks == 1 ){
      blocksize = max(xdim, ydim);
    }
    
    dim3 dimBlock(blocksize, blocksize, 1);
    dim3 dimGrid(xblocks, yblocks, 1);

#ifdef DEBUG
    cout << " lowEC: " << loEC << " hiEC " << hiEC << endl;
    cout << " lo   : " << lo   << " hi:  " << hi << endl;
    cout << " xdim: " << xdim << " ydim: " << ydim << endl;
    cout << " blocksize: " << blocksize << " xblocks: " << xblocks << " yblocks: " << yblocks << endl;
#endif

    RT_flags.nRaySteps = 0;

#if 0
    //__________________________________
    // set up and launch kernel
    launchRayTraceKernel< T >(dimGrid, 
                              dimBlock,
                              d_matl,
                              patchP,
                              (cudaStream_t*)stream,
                              RT_flags,
                              labelNames,
                              abskg_gdw, 
                              sigmaT4_gdw, 
                              celltype_gdw, 
                              old_gdw, 
                              new_gdw);
 
 #endif                        
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

//______________________________________________________________________
//
template
void Ray::rayTraceGPU< float > (Task::CallBackEvent,
                                const ProcessorGroup*,
                                const PatchSubset*,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse*,
                                void* stream,
                                int deviceID,
                                bool,
                                Task::WhichDW,
                                Task::WhichDW,
                                Task::WhichDW,
                                const int  );
                                
template
void Ray::rayTraceGPU< double > (Task::CallBackEvent,
                                 const ProcessorGroup*,
                                 const PatchSubset*,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse*,
                                 void* stream,
                                 int deviceID,
                                 bool,
                                 Task::WhichDW,
                                 Task::WhichDW,
                                 Task::WhichDW,
                                 const int  );
