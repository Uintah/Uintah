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
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/DbgOutput.h>
#ifdef HAVE_CUDA
  #include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#endif
#include <iostream>

#define BLOCKSIZE 16
//#define PRINTF            // if using printf statements to debug

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAYGPU", false);

//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup for ray trace kernel
//---------------------------------------------------------------------------
template<class T>
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
  if (event == Task::GPU) {
#ifdef HAVE_CUDA

    if (doCarryForward(radCalc_freq)) {
      return;
    }

    //__________________________________
    //  increase the size of the printbuffer on the device
#ifdef PRINTF
    size_t size;
    cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*size );
    printf(" Increasing the size of the print buffer from %lu to %lu bytes\n",(long uint) size, ((long uint)10 * size) );
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
    varLabelNames*  labelNames = new varLabelNames;

    labelNames->abskg   = d_abskgLabel->getName().c_str();    // CUDA doesn't support C++ strings
    labelNames->sigmaT4 = d_sigmaT4Label->getName().c_str();
    labelNames->divQ    = d_divQLabel->getName().c_str();
    labelNames->celltype  = d_cellTypeLabel->getName().c_str();
    labelNames->boundFlux = d_boundFluxLabel->getName().c_str();
    labelNames->radVolQ   = d_radiationVolqLabel->getName().c_str();
    
    labelNames->print();

    //__________________________________
    //  RMCRT_flags
    RMCRT_flags RT_flags;
    RT_flags.modifies_divQ = modifies_divQ;

    RT_flags.solveDivQ = d_solveDivQ;
    RT_flags.allowReflect = d_allowReflect;
    RT_flags.solveBoundaryFlux = d_solveBoundaryFlux;
    RT_flags.CCRays = d_CCRays;
    RT_flags.usingFloats = (d_FLT_DBL == TypeDescription::float_type);

    RT_flags.sigma = d_sigma;
    RT_flags.sigmaScat = d_sigmaScat;
    RT_flags.threshold = d_threshold;

    RT_flags.nDivQRays = d_nDivQRays;
    RT_flags.nFluxRays = d_nFluxRays;

    double start = clock();

    //______________________________________________________________________
    //
    // patch loop
    int numPatches = patches->size();
    for (int p = 0; p < numPatches; ++p) {

      const Patch* patch = patches->get(p);
      printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

      // Calculate the memory block size
      const IntVector loEC = patch->getExtraCellLowIndex();
      const IntVector lo = patch->getCellLowIndex();

      const IntVector hiEC = patch->getExtraCellHighIndex();
      const IntVector hi = patch->getCellHighIndex();
      const IntVector patchSize = hiEC - loEC;

      const int xdim = patchSize.x();
      const int ydim = patchSize.y();
      const int zdim = patchSize.z();

      // get the cell spacing and convert patch extents to CUDA vector type
      patchParams patchP;
      const Vector dx = patch->dCell();
      patchP.dx = make_double3(dx.x(), dx.y(), dx.z());
      patchP.lo = make_int3(lo.x(), lo.y(), lo.z());
      patchP.hi = make_int3(hi.x(), hi.y(), hi.z());

      patchP.loEC = make_int3(loEC.x(), loEC.y(), loEC.z());
      patchP.hiEC = make_int3(hiEC.x(), hiEC.y(), hiEC.z());

      patchP.ID = patch->getID();
      patchP.nCells = make_int3(xdim, ydim, zdim);

      // define dimensions of the thread grid to be launched
      int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
      int yblocks = (int)ceil((float)ydim / BLOCKSIZE);

      // if the # cells in a block < BLOCKSIZE^2 reduce block size
      int blocksize = BLOCKSIZE;
      if (xblocks == 1 && yblocks == 1) {
        blocksize = std::max(xdim, ydim);
      }

      dim3 dimBlock(blocksize, blocksize, 1);
      dim3 dimGrid(xblocks, yblocks, 1);

#ifdef DEBUG
      cout << " lowEC: " << loEC << " hiEC " << hiEC << endl;
      cout << " lo   : " << lo << " hi:  " << hi << endl;
      cout << " xdim: " << xdim << " ydim: " << ydim << endl;
      cout << " blocksize: " << blocksize << " xblocks: " << xblocks << " yblocks: " << yblocks << endl;
#endif

      RT_flags.nRaySteps = 0;

      //__________________________________
      // set up and launch kernel
      launchRayTraceKernel<T>(dimGrid,
                              dimBlock,
                              d_matl,
                              patchP,
                              (cudaStream_t*)stream,
                              RT_flags, labelNames, abskg_gdw,
                              sigmaT4_gdw,
                              celltype_gdw,
                              old_gdw,
                              new_gdw);

      //__________________________________
      //
      double end = clock();
      double efficiency = RT_flags.nRaySteps / ((end - start) / CLOCKS_PER_SEC);

      if (patch->getGridIndex() == 0) {
        std::cout << "\n";
        std::cout << " RMCRT REPORT: Patch 0" << "\n";
        std::cout << " Used " << (end - start) * 1000 / CLOCKS_PER_SEC << " milliseconds of CPU time. \n" << "\n";  // Convert time to ms
        std::cout << " Size: " << RT_flags.nRaySteps << "\n";
        std::cout << " Efficiency: " << efficiency << " steps per sec" << "\n";
        std::cout << std::endl;
      }
    }  //end patch loop
#endif // end #ifdef HAVE_CUDA
  }  //end GPU task code
}  // end GPU ray trace method

//---------------------------------------------------------------------------
// Method: The GPU data onion ray tracer - setup for ray trace data onion kernel
//---------------------------------------------------------------------------
template<class T>
void Ray::rayTraceDataOnionGPU( Task::CallBackEvent event,
                               const ProcessorGroup* pg,
                               const PatchSubset* finePatches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               void* stream,
                               int deviceID,
                               bool modifies_divQ,
                               Task::WhichDW which_abskg_dw,
                               Task::WhichDW which_sigmaT4_dw,
                               Task::WhichDW which_celltype_dw,
                               const int radCalc_freq )
{
  if (event == Task::GPU) {

#ifdef HAVE_CUDA

    //__________________________________
    //  Carry Forward (old_dw -> new_dw)
    if ( doCarryForward( radCalc_freq ) ) {
      printTask( finePatches, dbggpu, "Doing Ray::rayTrace_dataOnionGPU carryForward ( divQ )" );

      new_dw->transferFrom( old_dw, d_divQLabel,          finePatches, matls, true );
      new_dw->transferFrom( old_dw, d_radiationVolqLabel, finePatches, matls, true );
      return;
    }

    //__________________________________
    //  increase the size of the printbuffer on the device
#ifdef PRINTF
    size_t size;
    cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*size );
    printf(" Increasing the size of the print buffer from %lu to %lu bytes\n",(long uint) size, ((long uint)10 * size) );
#endif

    //__________________________________
    //  bulletproofing   FIX ME 
    const Level* fineLevel = getLevel(finePatches);
    const int maxLevels   = fineLevel->getGrid()->numLevels();
    if ( maxLevels > d_MAXLEVELS) {
      ostringstream warn;
      warn << "\nERROR:  RMCRT:GPU The maximum number of levels allowed ("<<d_MAXLEVELS<< ") has been exceeded." << endl;
      warn << " To increase that value see /src/CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh \n";
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    //  Grid Parameters
    gridParams gridP;
//    const Level* fineLevel = getLevel(finePatches);
//    const int maxLevels   = fineLevel->getGrid()->numLevels();
    gridP.maxLevels = maxLevels;
    LevelP level_0  = new_dw->getGrid()->getLevel(0);

    // Determine the size of the domain.
    BBox domain_BB;
    level_0->getInteriorSpatialRange( domain_BB );                 // edge of computational domain
    Point lo = domain_BB.min();
    Point hi = domain_BB.max();
    gridP.domain_BB.lo = make_double3( lo.x(), lo.y(), lo.z() );
    gridP.domain_BB.hi = make_double3( hi.x(), hi.y(), hi.z() );

    //__________________________________
    //  Level Parameters - first batch of level data
    levelParams levelP[maxLevels];
    for (int l = 0; l < maxLevels; ++l) {
      LevelP level = new_dw->getGrid()->getLevel(l);      
      levelP[l].hasFinerLevel = level->hasFinerLevel();

      SCIRun::Vector dx = level->dCell();
      levelP[l].Dx    = GPUVector(make_double3(dx.x(), dx.y(), dx.z()));
      levelP[l].DyDx  = dx.y() / dx.x();
      levelP[l].DzDx  = dx.z() / dx.x();
      levelP[l].index = level->getIndex();
      Point anchor    = level->getAnchor();
      levelP[l].anchor = GPUPoint( make_double3(anchor.x(), anchor.y(), anchor.z()));
      IntVector RR = level->getRefinementRatio();
      levelP[l].refinementRatio = GPUIntVector( make_int3(RR.x(), RR.y(), RR.z() ) );
    }

    //__________________________________
    //   Assign dataWarehouses
    GPUDataWarehouse* old_gdw = old_dw->getGPUDW()->getdevice_ptr();
    GPUDataWarehouse* new_gdw = new_dw->getGPUDW()->getdevice_ptr();

    GPUDataWarehouse* abskg_gdw    = new_dw->getOtherDataWarehouse(which_abskg_dw)->getGPUDW()->getdevice_ptr();
    GPUDataWarehouse* sigmaT4_gdw  = new_dw->getOtherDataWarehouse(which_sigmaT4_dw)->getGPUDW()->getdevice_ptr();
    GPUDataWarehouse* celltype_gdw = new_dw->getOtherDataWarehouse(which_celltype_dw)->getGPUDW()->getdevice_ptr();    
    
    //__________________________________
    //  RMCRT_flags
    RMCRT_flags RT_flags;
    RT_flags.modifies_divQ     = modifies_divQ;

    RT_flags.solveDivQ         = d_solveDivQ;
    RT_flags.allowReflect      = d_allowReflect;
    RT_flags.solveBoundaryFlux = d_solveBoundaryFlux;
    RT_flags.CCRays = d_CCRays;
    RT_flags.usingFloats = (d_FLT_DBL == TypeDescription::float_type);

    RT_flags.sigma             = d_sigma;
    RT_flags.sigmaScat         = d_sigmaScat;
    RT_flags.threshold         = d_threshold;

    RT_flags.nDivQRays         = d_nDivQRays;
    RT_flags.nFluxRays         = d_nFluxRays;
    RT_flags.whichROI_algo     = d_whichROI_algo;

    double start = clock();
        
    //______________________________________________________________________
    //
    // patch loop - not really necessary for now, should be only one patch in the PatchSubset
    int numPatches = finePatches->size();
    for (int p = 0; p < numPatches; ++p) {

      const Patch* finePatch = finePatches->get(p);
      printTask(finePatches, finePatch, dbggpu, "Doing Ray::rayTraceGPU");
      
      IntVector ROI_Lo = IntVector(-9,-9,-9);
      IntVector ROI_Hi = IntVector(-9,-9,-9);
      std::vector<IntVector> regionLo(maxLevels);
      std::vector<IntVector> regionHi(maxLevels);
  
      //__________________________________
      // compute ROI the extents for "dynamic", "fixed" and "patch_based" ROI    
      computeExtents(level_0, fineLevel, finePatch, maxLevels, new_dw, ROI_Lo, ROI_Hi, regionLo,  regionHi);
      
      // move everything into GPU vars
      GPUIntVector fineLevel_ROI_Lo = GPUIntVector( make_int3(ROI_Lo.x(), ROI_Lo.y(), ROI_Lo.z()) );
      GPUIntVector fineLevel_ROI_Hi = GPUIntVector( make_int3(ROI_Hi.x(), ROI_Hi.y(), ROI_Hi.z()) );
      
      for (int l = 0; l < maxLevels; ++l) {
        IntVector rlo = regionLo[l];
        IntVector rhi = regionHi[l];
        levelP[l].regionLo = GPUIntVector(make_int3(rlo.x(), rlo.y(), rlo.z()));
        levelP[l].regionHi = GPUIntVector(make_int3(rhi.x(), rhi.y(), rhi.z()));
      }

      //
      // Calculate the memory block size
      const IntVector loEC = finePatch->getExtraCellLowIndex();
      const IntVector lo   = finePatch->getCellLowIndex();
      const IntVector hiEC = finePatch->getExtraCellHighIndex();
      const IntVector hi   = finePatch->getCellHighIndex();

      const IntVector patchSize = hiEC - loEC;

      const int xdim = patchSize.x();
      const int ydim = patchSize.y();
      const int zdim = patchSize.z();

      // get the cell spacing and convert patch extents to CUDA vector type
      patchParams patchP;
      const Vector dx = finePatch->dCell();
      patchP.dx = make_double3(dx.x(), dx.y(), dx.z());
      patchP.lo = make_int3(lo.x(), lo.y(), lo.z());
      patchP.hi = make_int3(hi.x(), hi.y(), hi.z());

      patchP.loEC = make_int3(loEC.x(), loEC.y(), loEC.z());
      patchP.hiEC = make_int3(hiEC.x(), hiEC.y(), hiEC.z());

      patchP.ID = finePatch->getID();
      patchP.nCells = make_int3(xdim, ydim, zdim);

      // define dimensions of the thread grid to be launched
      int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
      int yblocks = (int)ceil((float)ydim / BLOCKSIZE);

      // if the # cells in a block < BLOCKSIZE^2 reduce block size
      int blocksize = BLOCKSIZE;
      if (xblocks == 1 && yblocks == 1) {
        blocksize = std::max(xdim, ydim);
      }

      dim3 dimBlock(blocksize, blocksize, 1);
      dim3 dimGrid(xblocks, yblocks, 1);

#ifdef DEBUG
      cout << " lowEC: " << loEC << " hiEC " << hiEC << endl;
      cout << " lo   : " << lo << " hi:  " << hi << endl;
      cout << " xdim: " << xdim << " ydim: " << ydim << endl;
      cout << " blocksize: " << blocksize << " xblocks: " << xblocks << " yblocks: " << yblocks << endl;
#endif

      RT_flags.nRaySteps = 0;


      //__________________________________
      // set up and launch kernel
      launchRayTraceDataOnionKernel<T>(dimGrid,
                                       dimBlock,
                                       d_matl,
                                       patchP,
                                       gridP,
                                       levelP,
                                       fineLevel_ROI_Lo,
                                       fineLevel_ROI_Hi,
                                       (cudaStream_t*)stream,
                                       RT_flags, 
                                       abskg_gdw,
                                       sigmaT4_gdw,
                                       celltype_gdw,
                                       old_gdw,
                                       new_gdw);

      //__________________________________
      //
      double end = clock();
      double efficiency = RT_flags.nRaySteps / ((end - start) / CLOCKS_PER_SEC);

      // THIS DOESNT WORK:  nRaySteps is not defined
      if (finePatch->getGridIndex() == 0) {
        std::cout << "\n";
        std::cout << " RMCRT REPORT: Patch 0" << "\n";
        std::cout << " Used " << (end - start) * 1000 / CLOCKS_PER_SEC << " milliseconds of CPU time. \n" << "\n";  // Convert time to ms
        std::cout << " Size: " << RT_flags.nRaySteps << "\n";
        std::cout << " Efficiency: " << efficiency << " steps per sec" << "\n";
        std::cout << std::endl;
      }
    }  //end patch loop

#endif // end #ifdef HAVE_CUDA
  }  //end GPU task code
}

//______________________________________________________________________
//  Explicit template instantiations
template
void Ray::rayTraceGPU< float > ( Task::CallBackEvent,
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
                                 const int );

template
void Ray::rayTraceGPU< double > ( Task::CallBackEvent,
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
                                  const int );

template
void Ray::rayTraceDataOnionGPU< float > ( Task::CallBackEvent,
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
                                          const int );

template
void Ray::rayTraceDataOnionGPU< double > ( Task::CallBackEvent,
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
                                           const int );
