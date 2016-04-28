/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Util/GPU.h>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/uintah_defs.h>
#include <curand.h>
#include <curand_kernel.h>

#define DEBUG -9                 // 1: divQ, 2: boundFlux, 3: scattering
#define FIXED_RANDOM_NUM         // also edit in src/Core/Math/MersenneTwister.h to compare with Ray:CPU

//__________________________________
//  To Do
//  - Add rayDirectionHyperCube_cellFace()
//  - Need to implement transferFrom so use can use calc_frequency > 1
//  - Temporal scheduling
//  - restarts are not working.
//  - Investigate using multiple GPUs per node.
//  - Implement fixed and dynamic ROI.
//  - dynamic block size?
//  - Implement labelNames in unified memory.
//  - investigate the performance with different patch configurations
//  - deterministic random numbers
//  - Ray steps


//__________________________________
//
//  To use cuda-gdb on a single GPU you must set the environmental variable
//  CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
//
// mpirun -np 1 xterm -e cuda-gdb sus -gpu -nthreads 2 <args>
//__________________________________

namespace Uintah {

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
template< class T>
__global__ void rayTraceKernel( dim3 dimGrid,
                                dim3 dimBlock,
                                const int matl,
                                levelParams level,
                                patchParams patch,
                                curandState* randNumStates,
                                RMCRT_flags RT_flags,
                                varLabelNames* labelNames,
                                GPUDataWarehouse* abskg_gdw,
                                GPUDataWarehouse* sigmaT4_gdw,
                                GPUDataWarehouse* cellType_gdw,
                                GPUDataWarehouse* old_gdw,
                                GPUDataWarehouse* new_gdw )
{

    // Not used right now
//  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x + patch.loEC.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y + patch.loEC.y;

  const GPUGridVariable< T > sigmaT4OverPi;
  const GPUGridVariable< T > abskg;              // Need to use getRegion() to get the data
  const GPUGridVariable<int> cellType;

  GPUGridVariable<double> divQ;
  GPUGridVariable<GPUStencil7> boundFlux;
  GPUGridVariable<double> radiationVolQ;

//  sigmaT4_gdw->print();

  sigmaT4_gdw->getLevel( sigmaT4OverPi, "sigmaT4",  matl, level.index);
  cellType_gdw->getLevel( cellType,     "cellType", matl, level.index);

  if(RT_flags.usingFloats){
    abskg_gdw->getLevel( abskg, "abskgRMCRT",  matl, level.index);
  }else{
    abskg_gdw->getLevel( abskg, "abskg",       matl, level.index);
  }

  if( RT_flags.modifies_divQ ){
    new_gdw->getModifiable( divQ,         "divQ",           patch.ID, matl );
    new_gdw->getModifiable( boundFlux,    "RMCRTboundFlux", patch.ID, matl );
    new_gdw->getModifiable( radiationVolQ,"radiationVolq",  patch.ID, matl );
  }else{
    new_gdw->get( divQ,         "divQ",           patch.ID, matl );         // these should be allocateAndPut() calls
    new_gdw->get( boundFlux,    "RMCRTboundFlux", patch.ID, matl );
    new_gdw->get( radiationVolQ,"radiationVolq",  patch.ID, matl );


    // Extra Cell Loop
    if ( (tidX >= patch.loEC.x) && (tidY >= patch.loEC.y) && (tidX < patch.hiEC.x) && (tidY < patch.hiEC.y) ) { // patch boundary check
      #pragma unroll
      for (int z = patch.loEC.z; z < patch.hiEC.z; z++) { // loop through z slices
        GPUIntVector c = make_int3(tidX, tidY, z);
        divQ[c]          = 0.0;
        radiationVolQ[c] = 0.0;
      }
    }
  }

  //__________________________________
  //  Sanity checks
#if 0
  if (isThread0()) {
   printf("  GPUVariable Sanity check level: %i, patch: %i \n",level.index, patch.ID);
  }
#endif
  GPUVariableSanityCK(abskg,         patch.loEC, patch.hiEC);
  GPUVariableSanityCK(sigmaT4OverPi, patch.loEC, patch.hiEC);


  bool doLatinHyperCube = (RT_flags.rayDirSampleAlgo == LATIN_HYPER_CUBE);

  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  // TO BE FILLED IN

  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  if( RT_flags.solveBoundaryFlux ){

    __shared__ int3 dirIndexOrder[6];
    __shared__ int3 dirSignSwap[6];

    //_____________________________________________
    //   Ordering for Surface Method
    // This block of code is used to properly place ray origins, and orient ray directions
    // onto the correct face.  This is necessary, because by default, the rays are placed
    // and oriented onto a default face, then require adjustment onto the proper face.
    dirIndexOrder[EAST]   = make_int3(2, 1, 0);
    dirIndexOrder[WEST]   = make_int3(2, 1, 0);
    dirIndexOrder[NORTH]  = make_int3(0, 2, 1);
    dirIndexOrder[SOUTH]  = make_int3(0, 2, 1);
    dirIndexOrder[TOP]    = make_int3(0, 1, 2);
    dirIndexOrder[BOT]    = make_int3(0, 1, 2);

    // Ordering is slightly different from 6Flux since here, rays pass through origin cell from the inside faces.
    dirSignSwap[EAST]     = make_int3(-1, 1,  1);
    dirSignSwap[WEST]     = make_int3( 1, 1,  1);
    dirSignSwap[NORTH]    = make_int3( 1, -1, 1);
    dirSignSwap[SOUTH]    = make_int3( 1, 1,  1);
    dirSignSwap[TOP]      = make_int3( 1, 1, -1);
    dirSignSwap[BOT]      = make_int3( 1, 1,  1);

    //__________________________________
    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if ( (tidX >= patch.lo.x) && (tidY >= patch.lo.y) && (tidX < patch.hi.x) && (tidY < patch.hi.y) ) { // patch boundary check
      #pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices

        GPUIntVector origin = make_int3(tidX, tidY, z);  // for each thread

        boundFlux[origin].initialize(0.0);

        BoundaryFaces boundaryFaces;

         // which surrounding cells are boundaries
        boundFlux[origin].p = has_a_boundaryDevice(origin, cellType, boundaryFaces);
        
        GPUPoint CC_pos = level.getCellPosition(origin);

        //__________________________________
        // Loop over boundary faces of the cell and compute incident radiative flux
        #pragma unroll
        for( int i = 0; i<boundaryFaces.size(); i++) {

          int RayFace = boundaryFaces.faceArray[i];
          int UintahFace[6] = {WEST,EAST,SOUTH,NORTH,BOT,TOP};

          double sumI     = 0;
          double sumProjI = 0;
          double sumI_prev= 0;

          //__________________________________
          // Flux ray loop
          #pragma unroll
          for (int iRay=0; iRay < RT_flags.nFluxRays; iRay++){

            GPUVector direction_vector;
            GPUVector rayOrigin;
            double cosTheta;

//  Need to add rayDirectionHyperCube_cellFace Call  -Todd                     <<<<<<<<<<<<<<<<<<<<<<<<<,,,

            rayDirection_cellFaceDevice( randNumStates, origin, dirIndexOrder[RayFace], dirSignSwap[RayFace], iRay,
                                   direction_vector, cosTheta );

            rayLocation_cellFaceDevice( randNumStates, RayFace, patch.dx, CC_pos, rayOrigin);

            updateSumIDevice< T >( level, direction_vector, rayOrigin, origin, patch.dx, sigmaT4OverPi, abskg, cellType, sumI, randNumStates, RT_flags);

            sumProjI += cosTheta * (sumI - sumI_prev);   // must subtract sumI_prev, since sumI accumulates intensity

            sumI_prev = sumI;

          } // end of flux ray loop

          //__________________________________
          //  Compute Net Flux to the boundary
          int face = UintahFace[RayFace];
          boundFlux[origin][ face ] = sumProjI * 2 *M_PI/RT_flags.nFluxRays;

/*`==========TESTING==========*/
#if DEBUG == 2
          printf( "\n      [%d, %d, %d]  face: %d sumProjI:  %g BF: %g\n",
                    origin.x, origin.y, origin.z, face, sumProjI, boundFlux[origin][ face ]);
#endif
/*===========TESTING==========`*/

        } // boundary faces loop
      }  // z slices loop
    }  // X-Y Thread loop
  }


  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________
  if( RT_flags.solveDivQ ){
    const int nDivQRays = RT_flags.nDivQRays;               // for readability

    //int rand_i[ doLatinHyperCube ? nDivQRays : 0 ];                                        // only needed for LHC scheme
    const int size = 1000;                                         // FIX ME Todd
    int rand_i[ size ];                                      // FIX ME TODD

    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if ( (tidX >= patch.lo.x) && (tidY >= patch.lo.y) && (tidX < patch.hi.x) && (tidY < patch.hi.y) ) { // patch boundary check
      #pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) {       // loop through z slices

        GPUIntVector origin = make_int3(tidX, tidY, z);     // for each thread
        double sumI = 0;
        GPUPoint CC_pos = level.getCellPosition(origin);

        if (doLatinHyperCube){
          randVectorDevice(rand_i, size,randNumStates);
        }

        //__________________________________
        // ray loop
        #pragma unroll

        for (int iRay = 0; iRay < nDivQRays; iRay++) {

          GPUVector direction_vector;
          if ( doLatinHyperCube ){                          // Latin-Hyper-Cube sampling
            direction_vector = findRayDirectionHyperCubeDevice(randNumStates, nDivQRays, rand_i[iRay], iRay );
          }else{                                            // Naive Monte-Carlo sampling
            direction_vector = findRayDirectionDevice( randNumStates );
          }
          
          GPUVector rayOrigin = rayOriginDevice( randNumStates, CC_pos, patch.dx, RT_flags.CCRays );

          updateSumIDevice< T >( level, direction_vector, rayOrigin, origin, patch.dx,  sigmaT4OverPi, abskg, cellType, sumI, randNumStates, RT_flags);
        } //Ray loop

        //__________________________________
        //  Compute divQ
        divQ[origin] = -4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/RT_flags.nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
        radiationVolQ[origin] = 4.0 * M_PI * abskg[origin] *  (sumI/RT_flags.nDivQRays) ;

/*`==========TESTING==========*/
#if DEBUG == 1
        if( isDbgCellDevice( origin ) ){
          printf( "\n      [%d, %d, %d]  sumI: %1.16e  divQ: %1.16e radiationVolq: %1.16e  abskg: %1.16e,    sigmaT4: %1.16e \n",
                    origin.x, origin.y, origin.z, sumI,divQ[origin], radiationVolQ[origin],abskg[origin], sigmaT4OverPi[origin]);
        }
#endif
/*===========TESTING==========`*/
      }  // end z-slice loop
    }  // end domain boundary check
  }  // solve divQ
}  // end ray trace kernel

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer data onion kernel
//---------------------------------------------------------------------------
// hard-wired for 2-levels now, but this should be fast and fixes
__constant__ levelParams d_levels[d_MAXLEVELS];

template< class T>
__global__ void rayTraceDataOnionKernel( dim3 dimGrid,
                                         dim3 dimBlock,
                                         int matl,
                                         patchParams finePatch,
                                         gridParams gridP,
                                         GPUIntVector fineLevel_ROI_Lo,
                                         GPUIntVector fineLevel_ROI_Hi,
                                         int3* regionLo,
                                         int3* regionHi,
                                         curandState* randNumStates,
                                         RMCRT_flags RT_flags,
                                         GPUDataWarehouse* abskg_gdw,
                                         GPUDataWarehouse* sigmaT4_gdw,
                                         GPUDataWarehouse* cellType_gdw,
                                         GPUDataWarehouse* old_gdw,
                                         GPUDataWarehouse* new_gdw )
{

    // Not used right now
//  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;


  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x + finePatch.loEC.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y + finePatch.loEC.y;

#if 0
  if (tidX == 1 && tidY == 1) {
    printf("\nGPU levelParams\n");

    printf("Level-0 ");
    d_levels[0].print();

    printf("Level-1 ");
    d_levels[1].print();
  }
#endif


  int maxLevels = gridP.maxLevels;
  int fineL = maxLevels - 1;

  //__________________________________
  //
  const GPUGridVariable<T>    abskg[d_MAXLEVELS];
  const GPUGridVariable<T>    sigmaT4OverPi[d_MAXLEVELS];
  const GPUGridVariable<int>  cellType[d_MAXLEVELS];

//  new_gdw->print();

  //__________________________________
  // coarse level data for the entire level
  for (int l = 0; l < maxLevels; ++l) {
    if (d_levels[l].hasFinerLevel) {
      abskg_gdw->getLevel( abskg[l],           "abskg",    matl, l);
      sigmaT4_gdw->getLevel( sigmaT4OverPi[l], "sigmaT4",  matl, l);
      cellType_gdw->getLevel( cellType[l],     "cellType", matl, l);

      GPUVariableSanityCK(abskg[l],        d_levels[l].regionLo,d_levels[l].regionHi);
      GPUVariableSanityCK(sigmaT4OverPi[l],d_levels[l].regionLo,d_levels[l].regionHi);
    }
  }

  //__________________________________
  //  fine level data for the region of interest.
  //  ToDo:  replace get with getRegion() calls so
  //  so the halo can be > 0
  if ( RT_flags.whichROI_algo == patch_based ) {

    abskg_gdw->get(abskg[fineL],           "abskg",    finePatch.ID, matl, fineL);
    sigmaT4_gdw->get(sigmaT4OverPi[fineL], "sigmaT4",  finePatch.ID, matl, fineL);
    cellType_gdw->get(cellType[fineL],     "cellType", finePatch.ID, matl, fineL);

    GPUVariableSanityCK(abskg[fineL],        fineLevel_ROI_Lo,fineLevel_ROI_Hi);
    GPUVariableSanityCK(sigmaT4OverPi[fineL],fineLevel_ROI_Lo,fineLevel_ROI_Hi);
  }

  GPUGridVariable<double> divQ;
  GPUGridVariable<GPUStencil7> boundFlux;
  GPUGridVariable<double> radiationVolQ;

  //__________________________________
  //  fine level data for this patch
  if( RT_flags.modifies_divQ ){
    new_gdw->getModifiable( divQ,         "divQ",          finePatch.ID, matl, fineL );
    new_gdw->getModifiable( boundFlux,    "boundFlux",     finePatch.ID, matl, fineL );
    new_gdw->getModifiable( radiationVolQ,"radiationVolq", finePatch.ID, matl, fineL );
  }else{
    new_gdw->get( divQ,         "divQ",           finePatch.ID, matl, fineL );         // these should be allocateAntPut() calls
    new_gdw->get( boundFlux,    "RMCRTboundFlux", finePatch.ID, matl, fineL );
    new_gdw->get( radiationVolQ,"radiationVolq",  finePatch.ID, matl, fineL );


    //__________________________________
    // initialize Extra Cell Loop
    if ( (tidX >= finePatch.loEC.x) && (tidY >= finePatch.loEC.y) && (tidX < finePatch.hiEC.x) && (tidY < finePatch.hiEC.y) ) { // finePatch boundary check
      #pragma unroll
      for (int z = finePatch.loEC.z; z < finePatch.hiEC.z; z++) { // loop through z slices
        GPUIntVector c = make_int3(tidX, tidY, z);
        divQ[c]          = 0.0;
        radiationVolQ[c] = 0.0;
      }
    }
  }

  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  // TO BE FILLED IN



  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  if( RT_flags.solveBoundaryFlux ){
    // TO BE FILLED IN
  }


#if 1
  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________
  if( RT_flags.solveDivQ ) {

    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if ( (tidX >= finePatch.lo.x) && (tidY >= finePatch.lo.y) && (tidX < finePatch.hi.x) && (tidY < finePatch.hi.y) ) { // finePatch boundary check
      #pragma unroll
      for (int z = finePatch.lo.z; z < finePatch.hi.z; z++) { // loop through z slices

        GPUIntVector origin = make_int3(tidX, tidY, z);  // for each thread

/*`==========TESTING==========*/
#if 0
        if( !isDbgCellDevice( origin ) ){
          return;
        }
     printf(" origin[%i,%i,%i] finePatchID: %i \n", origin.x, origin.y, origin.z, finePatch.ID);
#endif
/*===========TESTING==========`*/

        double sumI = 0;

        //__________________________________
        // ray loop
        #pragma unroll
        for (int iRay = 0; iRay < RT_flags.nDivQRays; iRay++) {

          GPUVector ray_direction = findRayDirectionDevice( randNumStates );

          GPUVector rayOrigin = rayOriginDevice( randNumStates, origin, d_levels[fineL].DyDx, d_levels[fineL].DzDx , RT_flags.CCRays );

          updateSumI_MLDevice<T>(ray_direction, rayOrigin, origin, gridP,
                                 fineLevel_ROI_Lo, fineLevel_ROI_Hi,
                                 regionLo, regionHi,
                                 sigmaT4OverPi, abskg, cellType, sumI, randNumStates, RT_flags);
        } //Ray loop

        //__________________________________
        //  Compute divQ
        divQ[origin] = -4.0 * M_PI * abskg[fineL][origin] * ( sigmaT4OverPi[fineL][origin] - (sumI/RT_flags.nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used
        radiationVolQ[origin] = 4.0 * M_PI * abskg[fineL][origin] *  (sumI/RT_flags.nDivQRays) ;


/*`==========TESTING==========*/
#if DEBUG == 1
       if( isDbgCellDevice(origin) ){
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n",
                    origin.x, origin.y, origin.z, sumI,divQ[origin], radiationVolQ[origin],abskg[fineL][origin], sigmaT4OverPi[fineL][origin]);
       }
#endif
/*===========TESTING==========`*/

      }  // end z-slice loop
    }  // end ROI loop
  }  // solve divQ
#endif

}

//______________________________________________________________________
//
//______________________________________________________________________
__device__ GPUVector findRayDirectionDevice( curandState* randNumStates )
{
  // Random Points On Sphere
  // add fuzz to prevent infs in 1/dirVector calculation
  double plusMinus_one = 2.0 * randDblExcDevice( randNumStates ) - 1.0 + DBL_EPSILON;
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);             // Radius of circle at z
  double theta = 2.0 * M_PI * randDblExcDevice( randNumStates );    // Uniform betwen 0-2Pi

  GPUVector dirVector;
  dirVector.x = r*cos(theta);   // Convert to cartesian coordinates
  dirVector.y = r*sin(theta);
  dirVector.z = plusMinus_one;

  return dirVector;
}


//______________________________________________________________________
//
__device__ GPUVector findRayDirectionHyperCubeDevice(curandState* randNumStates,
                                                     const int nDivQRays,
                                                     const int bin_i,
                                                     const int bin_j)
{
  // Random Points On Sphere
  double plusMinus_one = 2.0 *(randDblExcDevice( randNumStates ) + (double) bin_i)/nDivQRays - 1.0;

  // Radius of circle at z
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);

  // Uniform betwen 0-2Pi
  double phi = 2.0 * M_PI * (randDblExcDevice( randNumStates ) + (double) bin_j)/nDivQRays;

  GPUVector dirVector;
  dirVector[0] = r*cos(phi);                       // Convert to cartesian
  dirVector[1] = r*sin(phi);
  dirVector[2] = plusMinus_one;

  return dirVector;
}
//______________________________________________________________________
//  Populate vector with integers which have been randomly shuffled.
//  This is sampling without replacement and can be used to in a
//  Latin-Hyper-Cube sampling scheme.  The algorithm used is the
//  modern Fisher-Yates shuffle.
//______________________________________________________________________
__device__ void randVectorDevice( int int_array[],
                                  const int size,
                                  curandState* randNumStates ){

  for (int i=0; i<size; i++){   // populate sequential array from 0 to size-1
    int_array[i] = i;
  }

  for (int i=size-1; i>0; i--){  // fisher-yates shuffle starting with size-1
    int rand_int =  randIntDevice(randNumStates, i);    // Random number between 0 & i
    int swap = int_array[i];
    int_array[i] = int_array[rand_int];
    int_array[rand_int] = swap;
  }
}
//______________________________________________________________________
// Compute the Ray direction from a cell face
__device__ void rayDirection_cellFaceDevice( curandState* randNumStates,
                                             const GPUIntVector& origin,
                                             const GPUIntVector& indexOrder,
                                             const GPUIntVector& signOrder,
                                             const int iRay,
                                             GPUVector& directionVector,
                                             double& cosTheta )
{
  // Surface Way to generate a ray direction from the positive z face
  double phi = 2 * M_PI * randDblDevice(randNumStates);  // azimuthal angle.  Range of 0 to 2pi
  double theta = acos(randDblDevice(randNumStates));     // polar angle for the hemisphere
  cosTheta = cos(theta);
  double sinTheta = sin(theta);

  //Convert to Cartesian
  GPUVector tmp;
  tmp[0] = sinTheta * cos(phi);
  tmp[1] = sinTheta * sin(phi);
  tmp[2] = cosTheta;

  // Put direction vector as coming from correct face,
  directionVector[0] = tmp[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmp[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmp[indexOrder[2]] * signOrder[2];
}


//______________________________________________________________________
//

// Used in dataOnion.  This will be removed soon.
__device__ GPUVector rayOriginDevice( curandState* randNumStates,
                                      const GPUIntVector origin,
                                      const double DyDx,
                                      const double DzDx,
                                      const bool useCCRays )
{
  GPUVector rayOrigin;
  if (useCCRays == false) {
    rayOrigin.x = (double)origin.x + randDblDevice(randNumStates);
    rayOrigin.y = (double)origin.y + randDblDevice(randNumStates) * DyDx;
    rayOrigin.z = (double)origin.z + randDblDevice(randNumStates) * DzDx;
  }
  else {
    rayOrigin.x = origin.x + 0.5;
    rayOrigin.y = origin.y + 0.5 * DyDx;
    rayOrigin.z = origin.z + 0.5 * DzDx;
  }
  return rayOrigin;
}

__device__ 
GPUVector rayOriginDevice( curandState* randNumStates,
                           const GPUPoint  CC_pos,
                           const GPUVector dx,
                           const bool   useCCRays)
{
  GPUVector rayOrigin;
  if( useCCRays == false ){
    rayOrigin[0] =  CC_pos.x - 0.5*dx.x  + randDblDevice(randNumStates) * dx.x; 
    rayOrigin[1] =  CC_pos.y - 0.5*dx.y  + randDblDevice(randNumStates) * dx.y; 
    rayOrigin[2] =  CC_pos.z - 0.5*dx.z  + randDblDevice(randNumStates) * dx.z;
  }else{
    rayOrigin[0] = CC_pos.x;
    rayOrigin[1] = CC_pos.y;
    rayOrigin[2] = CC_pos.z;
  }
  return rayOrigin;
}

//______________________________________________________________________
//  Compute the Ray location from a cell face
__device__ void rayLocation_cellFaceDevice( curandState* randNumStates,
                                            const GPUIntVector& origin,
                                            const GPUIntVector &indexOrder,
                                            const GPUIntVector &shift,
                                            const double &DyDx,
                                            const double &DzDx,
                                            GPUVector& location )
{
  GPUVector tmp;
  tmp[0] = randDblDevice(randNumStates);
  tmp[1] = 0;
  tmp[2] = randDblDevice(randNumStates) * DzDx;

  // Put point on correct face
  location[0] = tmp[indexOrder[0]] + (double)shift[0];
  location[1] = tmp[indexOrder[1]] + (double)shift[1] * DyDx;
  location[2] = tmp[indexOrder[2]] + (double)shift[2] * DzDx;

  location[0] += (double)origin.x;
  location[1] += (double)origin.y;
  location[2] += (double)origin.z;
}
//______________________________________________________________________
//
//  Compute the Ray location on a cell face
__device__ void rayLocation_cellFaceDevice( curandState* randNumStates,
                                            const int face,
                                            const GPUVector Dx,
                                            const GPUPoint CC_pos,
                                            GPUVector& rayOrigin)
{
  double cellOrigin[3];
  // left, bottom, back corner of the cell
  cellOrigin[X] = CC_pos.x - 0.5 * Dx[X];
  cellOrigin[Y] = CC_pos.y - 0.5 * Dx[Y];
  cellOrigin[Z] = CC_pos.z - 0.5 * Dx[Z];

  switch(face)
  {
    case WEST:
      rayOrigin[X] = cellOrigin[X];
      rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
      break;
    case EAST:
      rayOrigin[X] = cellOrigin[X] +  Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
      break;
    case SOUTH:
      rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
      rayOrigin[Y] = cellOrigin[Y];
      rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
      break;
    case NORTH:
      rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + randDblDevice(randNumStates) * Dx[Z];
      break;
    case BOT:
      rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z];
      break;
    case TOP:
      rayOrigin[X] = cellOrigin[X] + randDblDevice(randNumStates) * Dx[X];
      rayOrigin[Y] = cellOrigin[Y] + randDblDevice(randNumStates) * Dx[Y];
      rayOrigin[Z] = cellOrigin[Z] + Dx[Z];
      break;
    default:
//      throw InternalError("Ray::rayLocation_cellFace,  Invalid FaceType Specified", __FILE__, __LINE__);
      return;
  }
}
//______________________________________________________________________
//
__device__ bool has_a_boundaryDevice(const GPUIntVector &c,
                                     const GPUGridVariable<int>& celltype,
                                     BoundaryFaces &boundaryFaces){

  GPUIntVector adj = c;
  bool hasBoundary = false;

  adj[0] = c[0] - 1;     // west

  if ( celltype[adj]+1 ){              // cell type of flow is -1, so when cellType+1 isn't false, we
    boundaryFaces.addFace( WEST );     // know we're at a boundary
    hasBoundary = true;
  }

  adj[0] += 2;           // east

  if ( celltype[adj]+1 ){
    boundaryFaces.addFace( EAST );
    hasBoundary = true;
  }

  adj[0] -= 1;
  adj[1] = c[1] - 1;     // south

  if ( celltype[adj]+1 ){
    boundaryFaces.addFace( SOUTH );
    hasBoundary = true;
  }

  adj[1] += 2;           // north

  if ( celltype[adj]+1 ){
    boundaryFaces.addFace( NORTH );
    hasBoundary = true;
  }

  adj[1] -= 1;
  adj[2] = c[2] - 1;     // bottom

  if ( celltype[adj]+1 ){
    boundaryFaces.addFace( BOT );
    hasBoundary = true;
  }

  adj[2] += 2;           // top

  if ( celltype[adj]+1 ){
    boundaryFaces.addFace( TOP );
    hasBoundary = true;
  }

  return (hasBoundary);
}


//______________________________________________________________________
//
__device__ void findStepSizeDevice(int step[],
                                   bool sign[],
                                   const GPUVector& inv_direction_vector)
{
  // get new step and sign
  for ( int d= 0; d<3; d++ ){

    if (inv_direction_vector[d]>0){
      step[d] = 1;
      sign[d] = 1;
    }else{
      step[d] = -1;
      sign[d] = 0;
    }
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
__device__ void raySignStepDevice(GPUVector& sign,
                                  int cellStep[],
                                  const GPUVector& inv_direction_vector)
{
  // get new step and sign
  for ( int d=0; d<3; d++){
    double me = copysign((double)1.0, inv_direction_vector[d]);  // +- 1
    
    sign[d] = max(0.0, me);    // 0, 1
    
    cellStep[d] = int(me);
  }
}


//______________________________________________________________________
//
__device__ bool containsCellDevice( GPUIntVector low,
                                    GPUIntVector high,
                                    GPUIntVector cell,
                                    const int dir)
{
  return  low[dir] <= cell[dir] &&
          high[dir] > cell[dir];
}

//______________________________________________________________________
//          // used by dataOnion it will be replaced
__device__ void reflect(double& fs,
                         GPUIntVector& cur,
                         GPUIntVector& prevCell,
                         const double abskg,
                         bool& in_domain,
                         int& step,
                         bool& sign,
                         double& ray_direction)
{
  fs = fs * (1 - abskg);

  //put cur back inside the domain
  cur = prevCell;
  in_domain = true;

  // apply reflection condition
  step *= -1;                // begin stepping in opposite direction
  sign = (sign==1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa
  ray_direction *= -1;
}


__device__ void reflect(double& fs,
                         GPUIntVector& cur,
                         GPUIntVector& prevCell,
                         const double abskg,
                         bool& in_domain,
                         int& step,
                         double& sign,
                         double& ray_direction)
{
  fs = fs * (1 - abskg);

  //put cur back inside the domain
  cur = prevCell;
  in_domain = true;

  // apply reflection condition
  step *= -1;                // begin stepping in opposite direction
  sign *= -1;
  ray_direction *= -1;
}


//______________________________________________________________________
template< class T >
__device__ void updateSumIDevice ( levelParams level,
                                   GPUVector& ray_direction,
                                   GPUVector& ray_origin,
                                   const GPUIntVector& origin,
                                   const GPUVector& Dx,
                                   const GPUGridVariable< T >& sigmaT4OverPi,
                                   const GPUGridVariable< T >& abskg,
                                   const GPUGridVariable<int>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags)

{


  GPUIntVector cur = origin;
  GPUIntVector prevCell = cur;
  // Step and sign for ray marching
  int step[3];                              // Gives +1 or -1 based on sign
  GPUVector sign;                           //   is 0 for negative ray direction

  GPUVector inv_ray_direction = 1.0/ray_direction;
/*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCellDevice(origin) ) {
    printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x, origin.y, origin.z,ray_direction.x, ray_direction.y, ray_direction.z, ray_origin.x, ray_origin.y, ray_origin.z);
  }
#endif
/*===========TESTING==========`*/

  raySignStepDevice(sign, step, ray_direction);


  GPUPoint CC_pos = level.getCellPosition(origin);
  
  // rayDx is the distance from bottom, left, back, corner of cell to ray
  GPUVector rayDx;
  rayDx[0] = ray_origin.x - ( CC_pos.x - 0.5*Dx.x );         // this can be consolidated using GPUVector
  rayDx[1] = ray_origin.y - ( CC_pos.y - 0.5*Dx.y );
  rayDx[2] = ray_origin.z - ( CC_pos.z - 0.5*Dx.z );
  
  GPUVector tMax;
  tMax.x = (sign.x * Dx.x - rayDx.x) * inv_ray_direction.x;
  tMax.y = (sign.y * Dx.y - rayDx.y) * inv_ray_direction.y;
  tMax.z = (sign.z * Dx.z - rayDx.z) * inv_ray_direction.z;

  //Length of t to traverse one cell
  GPUVector tDelta;
  tDelta   = Abs(inv_ray_direction) * Dx;

  //Initializes the following values for each ray
  bool in_domain     = true;
  double tMax_prev   = 0;
  double intensity   = 1.0;
  double fs          = 1.0;
  int nReflect       = 0;                 // Number of reflections
  double optical_thickness      = 0;
  double expOpticalThick_prev   = 1.0;
  double rayLength              = 0.0;
  GPUVector ray_location        = ray_origin;


#ifdef RAY_SCATTER
  double scatCoeff = RT_flags.sigmaScat;          //[m^-1]  !! HACK !! This needs to come from data warehouse
  if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff;
#endif

  //+++++++Begin ray tracing+++++++++++++++++++
  //Threshold while loop
  while ( intensity > RT_flags.threshold ){

    DIR dir = NONE;

    while (in_domain){

      prevCell = cur;
      double disMin = -9;          // Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      dir = NONE;
      if ( tMax.x < tMax.y ){        // X < Y
        if ( tMax.x < tMax.z ){      // X < Z
          dir = X;
        } else {
          dir = Z;
        }
      } else {
        if( tMax.y < tMax.z ){       // Y < Z
          dir = Y;
        } else {
          dir = Z;
        }
      }

      //__________________________________
      //  update marching variables
      cur[dir]  = cur[dir] + step[dir];
      disMin    = (tMax[dir] - tMax_prev);
      tMax_prev = tMax[dir];
      tMax[dir] = tMax[dir] + tDelta[dir];
      rayLength += disMin;

      ray_location.x = ray_location.x + (disMin  * ray_direction.x);
      ray_location.y = ray_location.y + (disMin  * ray_direction.y);
      ray_location.z = ray_location.z + (disMin  * ray_direction.z);

      in_domain = (celltype[cur]==-1);  //cellType of -1 is flow         HARDWIRED WARNING

      optical_thickness += abskg[prevCell]*disMin; 

      RT_flags.nRaySteps ++;
      
/*`==========TESTING==========*/
#if ( DEBUG >= 1 )
if( isDbgCellDevice(origin) ){
    printf( "            cur [%d,%d,%d] prev [%d,%d,%d] ", cur.x, cur.y, cur.z, prevCell.x, prevCell.y, prevCell.z);
    printf( " dir %d ", dir );
    printf( "tMax [%g,%g,%g] ",tMax.x,tMax.y, tMax.z);
    printf( "rayLoc [%g,%g,%g] ",ray_location.x,ray_location.y, ray_location.z);
    printf( "disMin %g tMax[dir]: %g tMax_prev: %g, Dx[dir]: %g\n",disMin, tMax[dir], tMax_prev, Dx[dir]);

    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevCell],  sigmaT4OverPi[prevCell]);
    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i\n",abskg[cur], sigmaT4OverPi[cur], celltype[cur] );
    printf( "            optical_thickkness %g \t rayLength: %g\n", optical_thickness, rayLength);
}
#endif

/*===========TESTING==========`*/


      //Eqn 3-15(see below reference) while
      //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevCell] * ( expOpticalThick_prev - expOpticalThick ) * fs;

      expOpticalThick_prev = expOpticalThick;


#ifdef RAY_SCATTER
      if ( (rayLength > scatLength) && in_domain){

        // get new scatLength for each scattering event
        scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff;

        ray_direction     = findRayDirectionDevice( randNumStates );

        inv_ray_direction = 1.0/ray_direction;

        // get new step and sign
        int stepOld = step[dir];
        raySignStepDevice( sign, step, ray_direction);

        // if sign[dir] changes sign, put ray back into prevCell (back scattering)
        // a sign change only occurs when the product of old and new is negative
        if( step[dir] * stepOld < 0 ){
          cur = prevCell;
        }
        
        GPUPoint CC_pos = level.getCellPosition(cur);
        
         // rayDx is the distance from bottom, left, back, corner of cell to ray
        rayDx[0] = ray_origin.x - ( CC_pos.x - 0.5*Dx.x );         // this can be consolidated using GPUVector
        rayDx[1] = ray_origin.y - ( CC_pos.y - 0.5*Dx.y );
        rayDx[2] = ray_origin.z - ( CC_pos.z - 0.5*Dx.z );

        tMax.x = (sign.x * Dx.x - rayDx.x) * inv_ray_direction.x;
        tMax.y = (sign.y * Dx.y - rayDx.y) * inv_ray_direction.y;
        tMax.z = (sign.z * Dx.z - rayDx.z) * inv_ray_direction.z;

        // Length of t to traverse one cell
        tDelta    = Abs(inv_ray_direction) * Dx;
        tMax_prev = 0;
        rayLength = 0;  // allow for multiple scattering events per ray

/*`==========TESTING==========*/
#if (DEBUG == 3)
        if( isDbgCellDevice( origin)  ){
          printf( "            Scatter: [%i, %i, %i], rayLength: %g, tmax: %g, %g, %g  tDelta: %g, %g, %g  ray_dir: %g, %g, %g\n",cur.x, cur.y, cur.z,rayLength, tMax[0], tMax[1], tMax[2], tDelta.x, tDelta.y , tDelta.z, ray_direction.x, ray_direction.y , ray_direction.z);
          printf( "                    dir: %i sign: [%g, %g, %g], step [%i, %i, %i] cur: [%i, %i, %i], prevCell: [%i, %i, %i]\n", dir, sign[0], sign[1], sign[2], step[0], step[1], step[2], cur[0], cur[1], cur[2], prevCell[0], prevCell[1], prevCell[2] );
          printf( "                    ray_location: [%g, %g, %g]\n", rayLocation[0], rayLocation[1], rayLocation[2] );
//          printf("                     rayDx         [%g, %g, %g]  CC_pos[%g, %g, %g]\n", rayDx[0], rayDx[1], rayDx[2], CC_pos.x, CC_pos.y, CC_pos.z);
        }
#endif
/*===========TESTING==========`*/

      }
#endif

    } //end domain while loop.  ++++++++++++++

    //  wall emission 12/15/11
    double wallEmissivity = abskg[cur];

    if (wallEmissivity > 1.0){       // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[cur] * intensity;

    intensity = intensity * fs;


    // when a ray reaches the end of the domain, we force it to terminate.
    if( !RT_flags.allowReflect ) intensity = 0;


/*`==========TESTING==========*/
#if DEBUG >0
if( isDbgCellDevice(origin) ){
    printf( "            cur [%d,%d,%d] intensity: %g expOptThick: %g, fs: %g allowReflect: %i \n",
            cur.x, cur.y, cur.z, intensity,  exp(-optical_thickness), fs, RT_flags.allowReflect );

}
__syncthreads();
#endif
/*===========TESTING==========`*/
    //__________________________________
    //  Reflections
    if ( (intensity > RT_flags.threshold) && RT_flags.allowReflect){
      reflect( fs, cur, prevCell, abskg[cur], in_domain, step[dir], sign[dir], ray_direction[dir]);
      ++nReflect;
    }

  }  // threshold while loop.
} // end of updateSumI function

//______________________________________________________________________
//  Multi-level
 template< class T>
 __device__ void updateSumI_MLDevice (  GPUVector& ray_direction,
                                        GPUVector& ray_location,
                                        const GPUIntVector& origin,
                                        gridParams gridP,
                                        const GPUIntVector& fineLevel_ROI_Lo,
                                        const GPUIntVector& fineLevel_ROI_Hi,
                                        const int3* regionLo,
                                        const int3* regionHi,
                                        const GPUGridVariable< T >* sigmaT4OverPi,
                                        const GPUGridVariable< T >* abskg,
                                        const GPUGridVariable<int>* cellType,
                                        double& sumI,
                                        curandState* randNumStates,
                                        RMCRT_flags RT_flags )
{
  /*`==========TESTING==========*/
#if DEBUG == 1
  if( isDbgCellDevice(origin) ) {
    printf("        A) updateSumI_ML: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x, origin.y, origin.z,ray_direction.x, ray_direction.y, ray_direction.z, ray_location.x, ray_location.y, ray_location.z);
  }
#endif
  /*===========TESTING==========`*/
  int maxLevels = gridP.maxLevels;   // for readability
  int L = maxLevels - 1;       // finest level
  int prevLev = L;

  GPUIntVector cur = origin;
  GPUIntVector prevCell = cur;
  // Step and sign for ray marching
  int step[3];                                          // Gives +1 or -1 based on sign
  bool sign[3];

  GPUVector inv_ray_direction = 1.0 / ray_direction;
  findStepSizeDevice(step, sign, inv_ray_direction);

  //__________________________________
  // define tMax & tDelta on all levels
  // go from finest to coarset level so you can compare
  // with 1L rayTrace results.
  GPUVector tMax;         // (mixing bools, ints and doubles)
  tMax.x = (origin.x + sign[0] - ray_location.x) * inv_ray_direction.x;
  tMax.y = (origin.y + sign[1] * d_levels[L].DyDx - ray_location.y) * inv_ray_direction.y;
  tMax.z = (origin.z + sign[2] * d_levels[L].DzDx - ray_location.z) * inv_ray_direction.z;

  GPUVector tDelta[d_MAXLEVELS];
  for (int Lev = maxLevels - 1; Lev > -1; Lev--) {
    //Length of t to traverse one cell
    tDelta[Lev].x = fabs(inv_ray_direction[0]);
    tDelta[Lev].y = fabs(inv_ray_direction[1]) * d_levels[Lev].DyDx;
    tDelta[Lev].z = fabs(inv_ray_direction[2]) * d_levels[Lev].DzDx;
  }

  //Initializes the following values for each ray
  bool in_domain = true;
  double tMax_prev = 0;
  double intensity = 1.0;
  double fs = 1.0;
  int nReflect = 0;                 // Number of reflections
  double optical_thickness = 0;
  double expOpticalThick_prev = 1.0;
  bool onFineLevel = true;

  //______________________________________________________________________
  //  Threshold  loop

  while (intensity > RT_flags.threshold) {

    DIR dir = NONE;

    while (in_domain) {

      prevCell = cur;
      prevLev = L;
      double disMin = -9;          // Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      if (tMax.x < tMax.y) {        // X < Y
        if (tMax.x < tMax.z) {      // X < Z
          dir = X;
        }
        else {
          dir = Z;
        }
      }
      else {
        if (tMax.y < tMax.z) {       // Y < Z
          dir = Y;
        }
        else {
          dir = Z;
        }
      }

      // next cell index and position
      cur[dir] = cur[dir] + step[dir];
      GPUVector dx_prev = d_levels[L].Dx;           //  Used to compute coarsenRatio
      //__________________________________
      // Logic for moving between levels
      //  - Currently you can only move from fine to coarse level
      //  - Don't jump levels if ray is at edge of domain

      GPUPoint pos = d_levels[L].getCellPosition(cur);         // position could be outside of domain
      in_domain = gridP.domain_BB.inside(pos);

      //in_domain = (cellType[L][cur] == d_flowCell);    // use this when direct comparison with 1L resullts

      bool ray_outside_ROI    = ( containsCellDevice(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir) == false );
      bool ray_outside_Region = ( containsCellDevice(regionLo[L], regionHi[L], cur, dir) == false );

      bool jumpFinetoCoarserLevel   = ( onFineLevel &&  ray_outside_ROI && in_domain );
      bool jumpCoarsetoCoarserLevel = ( (onFineLevel == false) && ray_outside_Region && (L > 0) && in_domain );

#if (DEBUG == 1 || DEBUG == 4)
      if( isDbgCellDevice(origin) ) {
        printf( "        Ray: [%i,%i,%i] **jumpFinetoCoarserLevel %i jumpCoarsetoCoarserLevel %i containsCell: %i ", cur.x, cur.y, cur.z, jumpFinetoCoarserLevel, jumpCoarsetoCoarserLevel,
            containsCellDevice(fineLevel_ROI_Lo, fineLevel_ROI_Hi, cur, dir));
        printf( " onFineLevel: %i ray_outside_ROI: %i ray_outside_Region: %i in_domain: %i\n", onFineLevel, ray_outside_ROI, ray_outside_Region,in_domain );
        printf( " L: %i regionLo: [%i,%i,%i], regionHi: [%i,%i,%i]\n",L,regionLo[L].x,regionLo[L].y,regionLo[L].z, regionHi[L].x,regionHi[L].y,regionHi[L].z);
      }
#endif

      if (jumpFinetoCoarserLevel) {
        cur = d_levels[L].mapCellToCoarser(cur);
        L = d_levels[L].getCoarserLevelIndex();      // move to a coarser level
        onFineLevel = false;

#if (DEBUG == 1 || DEBUG == 4)
        if( isDbgCellDevice(origin) ) {
          printf( "        ** Jumping off fine patch switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i] \n",prevLev, L, cur.x, cur.y, cur.z);
        }
#endif

      }
      else if (jumpCoarsetoCoarserLevel) {
        GPUIntVector c_old = cur;                     // needed for debugging
        cur = d_levels[L].mapCellToCoarser(cur);
        L = d_levels[L].getCoarserLevelIndex();      // move to a coarser level
#if (DEBUG == 1 || DEBUG == 4)
        if( isDbgCellDevice(origin) ) {
          printf( "        ** Switching Levels:  prev L: %i, L: %i, cur: [%i,%i,%i], c_old: [%i,%i,%i]\n",prevLev, L, cur.x, cur.y, cur.z, c_old.x, c_old.y, c_old.z);
        }
#endif
      }


      //__________________________________
      //  update marching variables
      disMin = (tMax[dir] - tMax_prev);
      tMax_prev = tMax[dir];
      tMax[dir] = tMax[dir] + tDelta[L][dir];

      ray_location.x = ray_location.x + (disMin * ray_direction.x);
      ray_location.y = ray_location.y + (disMin * ray_direction.y);
      ray_location.z = ray_location.z + (disMin * ray_direction.z);

      //__________________________________
      // Account for uniqueness of first step after reaching a new level
      GPUVector dx = d_levels[L].Dx;
      GPUIntVector coarsenRatio = GPUIntVector(make_int3(1, 1, 1));

      coarsenRatio[0] = dx[0] / dx_prev[0];
      coarsenRatio[1] = dx[1] / dx_prev[1];
      coarsenRatio[2] = dx[2] / dx_prev[2];

      GPUVector lineup;
      for (int ii = 0; ii < 3; ii++) {
        if (sign[ii]) {
          lineup[ii] = -(cur[ii] % coarsenRatio[ii] - (coarsenRatio[ii] - 1));
        }
        else {
          lineup[ii] = cur[ii] % coarsenRatio[ii];
        }
      }

      tMax += lineup * tDelta[prevLev];

      /*`==========TESTING==========*/
#if DEBUG == 1
      if( isDbgCellDevice(origin) ) {
        printf( "        B) cur [%i,%i,%i] prev [%i,%i,%i]"
            " dir %i "
            " stepSize [%i,%i,%i] "
            " tMax [%g,%g,%g] "
            "rayLoc [%g,%g,%g] "
            "inv_dir [%g,%g,%g] "
            "disMin %g "
            "inDomain %i\n"
            "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n"
            "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i \n"
            "            Dx[prevLev].x  %g \n",
            cur.x, cur.y, cur.z, prevCell.x, prevCell.y, prevCell.z,
            dir,
            step[0],step[1],step[2],
            tMax.x,tMax.y, tMax.z,
            ray_location.x,ray_location.y, ray_location.z,
            inv_ray_direction.x,inv_ray_direction.y, inv_ray_direction.z,
            disMin,
            in_domain,
            abskg[prevLev][prevCell], sigmaT4OverPi[prevLev][prevCell],
            abskg[L][cur], sigmaT4OverPi[L][cur], cellType[L][cur],
            d_levels[prevLev].Dx.x);
      }
#endif
      /*===========TESTING==========`*/
      optical_thickness += d_levels[prevLev].Dx.x * abskg[prevLev][prevCell] * disMin;

      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevLev][prevCell] * (expOpticalThick_prev - expOpticalThick) * fs;

      expOpticalThick_prev = expOpticalThick;

    }  //end domain while loop.  ++++++++++++++
    //__________________________________
    //
    double wallEmissivity = abskg[L][cur];

    if (wallEmissivity > 1.0) {       // Ensure wall emissivity doesn't exceed one.
      wallEmissivity = 1.0;
    }

    intensity = exp(-optical_thickness);

    sumI += wallEmissivity * sigmaT4OverPi[L][cur] * intensity;

    intensity = intensity * fs;

    // when a ray reaches the end of the domain, we force it to terminate.
    if (!RT_flags.allowReflect)
      intensity = 0;

    /*`==========TESTING==========*/
#if DEBUG == 1
    if( isDbgCellDevice(origin) ) {
      printf( "        C) intensity: %g OptThick: %g, fs: %g allowReflect: %i\n", intensity, optical_thickness, fs, RT_flags.allowReflect );
    }
#endif
    /*===========TESTING==========`*/
    //__________________________________
    //  Reflections
    if ((intensity > RT_flags.threshold) && RT_flags.allowReflect) {
      reflect(fs, cur, prevCell, abskg[L][cur], in_domain, step[dir], sign[dir], ray_direction[dir]);
      ++nReflect;
    }
  }  // threshold while loop.
}  // end of updateSumI function

//______________________________________________________________________
// Returns random number between 0 & 1.0 including 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________
__device__ double randDblDevice(curandState* globalState)
{
  int tid = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  curandState localState = globalState[tid];
  double val = curand(&localState);
  globalState[tid] = localState;

#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  return (double)val * (1.0/4294967295.0);
#endif

}

//______________________________________________________________________
// Returns random number between 0 & 1.0 excluding 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation
//______________________________________________________________________
__device__ double randDblExcDevice(curandState* globalState)
{
  int tid = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;

  curandState localState = globalState[tid];
  double val = curand(&localState);
  globalState[tid] = localState;

#ifdef FIXED_RANDOM_NUM
  return 0.3;
#else
  return ( (double)val + 0.5 ) * (1.0/4294967296.0);
#endif
}

//______________________________________________________________________
// Returns random integer in [0,n]
// rnd_integer_from_A_to_B = A + curand() * (B-A);
//  A = 0
//______________________________________________________________________
__device__ int randIntDevice(curandState* globalState,
                             const int B )
{
  double val = randDblDevice( globalState );
  return val * B;
}

//______________________________________________________________________
//  Each thread gets same seed, a different sequence number, no offset
//  This will create repeatable results.
__global__ void setupRandNumKernel(curandState* randNumStates)
{
  int tID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  curand_init(1234, tID, 0, &randNumStates[tID]);
}

//______________________________________________________________________
//  is cell a debug cell
__device__ bool isDbgCellDevice( GPUIntVector me )
{
  int size = 2;
  GPUIntVector dbgCell[2];
  dbgCell[0] = make_int3(0,0,0);
  dbgCell[1] = make_int3(5,5,5);



  for (int i = 0; i < size; i++) {
    if( me == dbgCell[i]){
      return true;
    }
  }
  return false;
}
//______________________________________________________________________
//   Perform some sanity checks on the Variable.  This is for debugging
template< class T>
__device__ void GPUVariableSanityCK(const GPUGridVariable<T>& Q,
                                    const GPUIntVector Lo,
                                    const GPUIntVector Hi)
{
#if SCI_ASSERTION_LEVEL > 0
  if (isThread0()) {
    GPUIntVector varLo = Q.getLowIndex();
    GPUIntVector varHi = Q.getHighIndex();

    if( Lo < varLo || varHi < Hi){
      printf ( "ERROR: GPUVariableSanityCK \n");
      printf("  Variable:          varLo:[%i,%i,%i], varHi[%i,%i,%i]\n", varLo.x, varLo.y, varLo.z, varHi.x, varHi.y, varHi.z);
      printf("  Requested extents: varLo:[%i,%i,%i], varHi[%i,%i,%i]\n", Lo.x, Lo.y, Lo.z, Hi.x, Hi.y, Hi.z);
      printf(" Now existing...");
      __threadfence();
      asm("trap;");
    }

    for (int i = Lo.x; i < Hi.x; i++) {
      for (int j = Lo.y; j < Hi.y; j++) {
        for (int k = Lo.z; k < Hi.z; k++) {
          GPUIntVector idx = make_int3(i, j, k);
          T me = Q[idx];
          if ( isnan(me) || isinf(me)){
            printf ( "isNan or isInf was detected at [%i,%i,%i]\n", i,j,k);
            printf(" Now existing...");
            __threadfence();
            asm("trap;");
          }

        }  // k loop
      }  // j loop
    }  // i loop
  }  // thread0
#endif
}
template
__device__ void GPUVariableSanityCK(const GPUGridVariable<float>& Q,
                                    const GPUIntVector Lo,
                                    const GPUIntVector Hi);
template
__device__ void GPUVariableSanityCK(const GPUGridVariable<double>& Q,
                                    const GPUIntVector Lo,
                                    const GPUIntVector Hi);
//______________________________________________________________________
//
template< class T>
__host__ void launchRayTraceKernel(dim3 dimGrid,
                                   dim3 dimBlock,
                                   const int matlIndx,
                                   levelParams level,
                                   patchParams patch,
                                   cudaStream_t* stream,
                                   RMCRT_flags RT_flags,
                                   varLabelNames* labelNames,
                                   GPUDataWarehouse* abskg_gdw,
                                   GPUDataWarehouse* sigmaT4_gdw,
                                   GPUDataWarehouse* cellType_gdw,
                                   GPUDataWarehouse* old_gdw,
                                   GPUDataWarehouse* new_gdw)
{
  // setup random number generator states on the device, 1 for each thread
  curandState* randNumStates;
  int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
  CUDA_RT_SAFE_CALL( cudaMalloc((void**)&randNumStates, numStates * sizeof(curandState)) );


  setupRandNumKernel<<< dimGrid, dimBlock, 0, *stream>>>( randNumStates );

  rayTraceKernel< T ><<< dimGrid, dimBlock, 0, *stream >>>( dimGrid,
                                                            dimBlock,
                                                            matlIndx,
                                                            level,
                                                            patch,
                                                            randNumStates,
                                                            RT_flags,
                                                            labelNames,
                                                            abskg_gdw,
                                                            sigmaT4_gdw,
                                                            cellType_gdw,
                                                            old_gdw,
                                                            new_gdw);
    // free device-side RNG states
    CUDA_RT_SAFE_CALL( cudaFree(randNumStates) );
}

//______________________________________________________________________
//
template< class T>
__host__ void launchRayTraceDataOnionKernel( dim3 dimGrid,
                                             dim3 dimBlock,
                                             int matlIndex,
                                             patchParams patch,
                                             gridParams gridP,
                                             levelParams* levelP,
                                             GPUIntVector fineLevel_ROI_Lo,
                                             GPUIntVector fineLevel_ROI_Hi,
                                             cudaStream_t* stream,
                                             RMCRT_flags RT_flags,
                                             GPUDataWarehouse* abskg_gdw,
                                             GPUDataWarehouse* sigmaT4_gdw,
                                             GPUDataWarehouse* cellType_gdw,
                                             GPUDataWarehouse* old_gdw,
                                             GPUDataWarehouse* new_gdw )
{
  // copy regionLo & regionHi to device memory
  int maxLevels = gridP.maxLevels;

  int3* dev_regionLo;
  int3* dev_regionHi;
  size_t size = d_MAXLEVELS *  sizeof(int3);
  CUDA_RT_SAFE_CALL( cudaMalloc( (void**)& dev_regionLo, size) );
  CUDA_RT_SAFE_CALL( cudaMalloc( (void**)& dev_regionHi, size) );

  int3 myLo[d_MAXLEVELS];
  int3 myHi[d_MAXLEVELS];
  for (int l = 0; l < maxLevels; ++l) {
    myLo[l] = levelP[l].regionLo;        // never use levelP regionLo or hi in the kernel.
    myHi[l] = levelP[l].regionHi;        // They are different on each patch
  }

  CUDA_RT_SAFE_CALL( cudaMemcpy( dev_regionLo, myLo, size, cudaMemcpyHostToDevice) );
  CUDA_RT_SAFE_CALL( cudaMemcpy( dev_regionHi, myHi, size, cudaMemcpyHostToDevice) );


  //__________________________________
  // copy levelParams array to constant memory on device
  CUDA_RT_SAFE_CALL(cudaMemcpyToSymbol(d_levels, levelP, (maxLevels * sizeof(levelParams))));

  //__________________________________
  // setup random number generator states on the device, 1 for each thread
  curandState* randNumStates;
  int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
  CUDA_RT_SAFE_CALL( cudaMalloc((void**)&randNumStates, (numStates * sizeof(curandState))) );

  setupRandNumKernel<<< dimGrid, dimBlock, 0, *stream>>>( randNumStates );

  rayTraceDataOnionKernel< T ><<< dimGrid, dimBlock, 0, *stream >>>( dimGrid,
                                                                     dimBlock,
                                                                     matlIndex,
                                                                     patch,
                                                                     gridP,
                                                                     fineLevel_ROI_Lo,
                                                                     fineLevel_ROI_Hi,
                                                                     dev_regionLo,
                                                                     dev_regionHi,
                                                                     randNumStates,
                                                                     RT_flags,
                                                                     abskg_gdw,
                                                                     sigmaT4_gdw,
                                                                     cellType_gdw,
                                                                     old_gdw,
                                                                     new_gdw);

  // free device-side RNG states
  CUDA_RT_SAFE_CALL( cudaFree(randNumStates) );
  CUDA_RT_SAFE_CALL( cudaFree(dev_regionLo) );
  CUDA_RT_SAFE_CALL( cudaFree(dev_regionHi) );

}

//______________________________________________________________________
//  Explicit template instantiations

template
__host__ void launchRayTraceKernel<double>( dim3 dimGrid,
                                            dim3 dimBlock,
                                            const int matlIndx,
                                            levelParams level,
                                            patchParams patch,
                                            cudaStream_t* stream,
                                            RMCRT_flags RT_flags,
                                            varLabelNames* labelNames,
                                            GPUDataWarehouse* abskg_gdw,
                                            GPUDataWarehouse* sigmaT4_gdw,
                                            GPUDataWarehouse* cellType_gdw,
                                            GPUDataWarehouse* old_gdw,
                                            GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template
__host__ void launchRayTraceKernel<float>( dim3 dimGrid,
                                           dim3 dimBlock,
                                           const int matlIndx,
                                           levelParams level,
                                           patchParams patch,
                                           cudaStream_t* stream,
                                           RMCRT_flags RT_flags,
                                           varLabelNames* labelNames,
                                           GPUDataWarehouse* abskg_gdw,
                                           GPUDataWarehouse* sigmaT4_gdw,
                                           GPUDataWarehouse* celltype_gdw,
                                           GPUDataWarehouse* old_gdw,
                                           GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template
__host__ void launchRayTraceDataOnionKernel<double>( dim3 dimGrid,
                                                     dim3 dimBlock,
                                                     int matlIndex,
                                                     patchParams patch,
                                                     gridParams gridP,
                                                     levelParams*  levelP,
                                                     GPUIntVector fineLevel_ROI_Lo,
                                                     GPUIntVector fineLevel_ROI_Hi,
                                                     cudaStream_t* stream,
                                                     RMCRT_flags RT_flags,
                                                     GPUDataWarehouse* abskg_gdw,
                                                     GPUDataWarehouse* sigmaT4_gdw,
                                                     GPUDataWarehouse* cellType_gdw,
                                                     GPUDataWarehouse* old_gdw,
                                                     GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template
__host__ void launchRayTraceDataOnionKernel<float>( dim3 dimGrid,
                                                    dim3 dimBlock,
                                                    int matlIndex,
                                                    patchParams patch,
                                                    gridParams gridP,
                                                    levelParams* levelP,
                                                    GPUIntVector fineLevel_ROI_Lo,
                                                    GPUIntVector fineLevel_ROI_Hi,
                                                    cudaStream_t* stream,
                                                    RMCRT_flags RT_flags,
                                                    GPUDataWarehouse* abskg_gdw,
                                                    GPUDataWarehouse* sigmaT4_gdw,
                                                    GPUDataWarehouse* cellType_gdw,
                                                    GPUDataWarehouse* old_gdw,
                                                    GPUDataWarehouse* new_gdw );

} //end namespace Uintah
