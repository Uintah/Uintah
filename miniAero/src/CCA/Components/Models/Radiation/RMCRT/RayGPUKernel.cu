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

#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <Core/Grid/Variables/GPUGridVariable.h>

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/GPUStencil7.h>
#include <sci_defs/cuda_defs.h>
#include <sci_defs/uintah_defs.h>
#include <curand.h>
#include <curand_kernel.h>

// TURN ON debug flag in src/Core/Math/MersenneTwister.h to compare with Ray:CPU
#define DEBUG -9 // 1: divQ, 2: boundFlux, 3: scattering
//#define FIXED_RANDOM_NUM

//__________________________________
//  To Do
//  - dynamic block size?
//  - use labelNames
//  - investigate the performance with different patch configurations
//  - deterministic random numbers
//  - Ray steps
//  - What's up with data onion and raylocation call?


namespace Uintah {

using namespace SCIRun;

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
template< class T>
__global__ void rayTraceKernel(dim3 dimGrid,
                               dim3 dimBlock,
                               int matl,
                               patchParams patch,
                               curandState* randNumStates,
                               RMCRT_flags RT_flags,
                               varLabelNames labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw)
{

  int blockID  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  
  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x + patch.loEC.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y + patch.loEC.y;
  
  const GPUGridVariable< T > sigmaT4OverPi;
  const GPUGridVariable< T > abskg;              // Need to use getRegion() to get the data
  const GPUGridVariable<int> celltype;

  GPUGridVariable<double> divQ;
  GPUGridVariable<GPUStencil7> boundFlux;
  GPUGridVariable<double> radiationVolQ;
 
    
  sigmaT4_gdw->get( sigmaT4OverPi , "sigmaT4",  patch.ID, matl );       // Should be using labelNames struct
  celltype_gdw->get( celltype,     "cellType", patch.ID, matl );
  
  if(RT_flags.usingFloats){
    abskg_gdw->get( abskg , "abskgRMCRT",   patch.ID, matl );
  }else{
    abskg_gdw->get( abskg , "abskg",   patch.ID, matl );
  }

  if( RT_flags.modifies_divQ ){
    new_gdw->getModifiable( divQ,         "divQ",          patch.ID, matl );
    new_gdw->getModifiable( boundFlux,    "boundFlux",     patch.ID, matl );
    new_gdw->getModifiable( radiationVolQ,"radiationVolq", patch.ID, matl );
  }else{
    new_gdw->get( divQ,         "divQ",          patch.ID, matl );         // these should be allocateAntPut() calls
    new_gdw->get( boundFlux,    "boundFlux",     patch.ID, matl );
    new_gdw->get( radiationVolQ,"radiationVolq", patch.ID, matl );
    
    
    // Extra Cell Loop
    if (tidX >= patch.loEC.x && tidY >= patch.loEC.y && tidX < patch.hiEC.x && tidY < patch.hiEC.y) { // patch boundary check
      #pragma unroll
      for (int z = patch.loEC.z; z < patch.hiEC.z; z++) { // loop through z slices
        gpuIntVector c = make_int3(tidX, tidY, z);
        divQ[c]          = 0.0;
        radiationVolQ[c] = 0.0;
      }
    }
  }
  

  
/*`==========TESTING==========*/
#if 0  
 //__________________________________
 // Sanity check code used to test the "iterators" 
  // Extra Cell Loop
  if (threadIdx.y == 2 ) {
    printf( "outside loops thread[%d, %d] tID[%d, %d]\n",threadIdx.x, threadIdx.y, tidX, tidY);
  }
  
  if (tidX >= patch.loEC.x && tidY >= patch.loEC.y && tidX < patch.hiEC.x && tidY < patch.hiEC.y) { // patch boundary check
    for (int z = patch.loEC.z; z < patch.hiEC.z; z++) { // loop through z slices
      gpuIntVector c = make_int3(tidX, tidY, z);
      divQ[c] = 0;

      if (c.y == 2 && c.z == 2 ) {
        printf( " EC thread[%d, %d] tID[%d, %d]\n",threadIdx.x, threadIdx.y, tidX, tidY);
      }
    }
  }  

  if (tidX >= patch.lo.x && tidY >= patch.lo.y && tidX < patch.hi.x && tidY < patch.hi.y) { // patch boundary check
    for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices
      gpuIntVector c = make_int3(tidX, tidY, z);
      if (c.y == 2 && c.z == 2 ) {
        printf( " int thread[%d, %d] tID[%d, %d]\n",threadIdx.x, threadIdx.y, tidX, tidY);
      }
      divQ[c] = c.x + c.y + c.z; 
    }
  }
  return;
#endif 
/*===========TESTING==========`*/

  double DyDx = patch.dx.y/patch.dx.x;
  double DzDx = patch.dx.z/patch.dx.x;
  
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
    __shared__ int3 locationIndexOrder[6];
    __shared__ int3 locationShift[6];
    
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

    locationIndexOrder[EAST]  = make_int3(1,0,2);
    locationIndexOrder[WEST]  = make_int3(1,0,2);
    locationIndexOrder[NORTH] = make_int3(0,1,2);
    locationIndexOrder[SOUTH] = make_int3(0,1,2);
    locationIndexOrder[TOP]   = make_int3(0,2,1);
    locationIndexOrder[BOT]   = make_int3(0,2,1);

    locationShift[EAST]   = make_int3(1, 0, 0);
    locationShift[WEST]   = make_int3(0, 0, 0);
    locationShift[NORTH]  = make_int3(0, 1, 0);
    locationShift[SOUTH]  = make_int3(0, 0, 0);
    locationShift[TOP]    = make_int3(0, 0, 1);
    locationShift[BOT]    = make_int3(0, 0, 0);  
    
    //__________________________________
    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if (tidX >= patch.lo.x && tidY >= patch.lo.y && tidX < patch.hi.x && tidY < patch.hi.y) { // patch boundary check
      #pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices
      
        gpuIntVector origin = make_int3(tidX, tidY, z);  // for each thread
        
        boundFlux[origin].initialize(0.0);

        BoundaryFaces boundaryFaces;

 /*
        if(RT_flags.benchMark==4 || RT_flags.benchMark==5){
          boundaryFaces.addFace(5);
        }
*/
        
        // which surrounding cells are boundaries
        boundFlux[origin].p = has_a_boundaryDevice(origin, celltype, boundaryFaces);

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

            gpuVector direction_vector, ray_location; 
            double cosTheta;

            rayDirection_cellFaceDevice( randNumStates, origin, dirIndexOrder[RayFace], dirSignSwap[RayFace], iRay,
                                   direction_vector, cosTheta );
                                   
            rayLocation_cellFaceDevice( randNumStates, origin, locationIndexOrder[RayFace], locationShift[RayFace], 
                                  DyDx, DzDx, ray_location);            

            updateSumIDevice< T >( direction_vector, ray_location, origin, patch.dx, sigmaT4OverPi, abskg, celltype, sumI, randNumStates, RT_flags);

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
    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if (tidX >= patch.lo.x && tidY >= patch.lo.y && tidX < patch.hi.x && tidY < patch.hi.y) { // patch boundary check
      #pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices

        gpuIntVector origin = make_int3(tidX, tidY, z);  // for each thread
        double sumI = 0;
        
        //__________________________________
        // ray loop
        #pragma unroll
        for (int iRay = 0; iRay < RT_flags.nDivQRays; iRay++) {
        
          gpuVector direction_vector = findRayDirectionDevice( randNumStates );
          
          gpuVector ray_location = rayLocationDevice( randNumStates, origin, DyDx,  DzDx, RT_flags.CCRays );
         
          updateSumIDevice< T >( direction_vector, ray_location, origin, patch.dx,  sigmaT4OverPi, abskg, celltype, sumI, randNumStates, RT_flags);
        } //Ray loop
 
        //__________________________________
        //  Compute divQ
        divQ[origin] = 4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/RT_flags.nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used 
        radiationVolQ[origin] = 4.0 * M_PI * abskg[origin] *  (sumI/RT_flags.nDivQRays) ;
        
/*`==========TESTING==========*/
#if DEBUG == 1
          printf( "\n      [%d, %d, %d]  sumI: %g  divQ: %g radiationVolq: %g  abskg: %g,    sigmaT4: %g \n", 
                    origin.x, origin.y, origin.z, sumI,divQ[origin], radiationVolQ[origin],abskg[origin], sigmaT4OverPi[origin]);
#endif
/*===========TESTING==========`*/
      }  // end z-slice loop
    }  // end domain boundary check
  }  // solve divQ
//}  // thread1
}  // end ray trace kernel

//______________________________________________________________________
//
//______________________________________________________________________
__device__ gpuVector findRayDirectionDevice( curandState* randNumStates )
{
  // Random Points On Sphere
  // add fuzz to prevent infs in 1/dirVector calculation
  double plusMinus_one = 2.0 * randDblExcDevice( randNumStates ) - 1.0 + DBL_EPSILON;
  double r = sqrt(1.0 - plusMinus_one * plusMinus_one);             // Radius of circle at z
  double theta = 2.0 * M_PI * randDblExcDevice( randNumStates );    // Uniform betwen 0-2Pi

  gpuVector dirVector;
  dirVector.x = r*cos(theta);   // Convert to cartesian coordinates
  dirVector.y = r*sin(theta);
  dirVector.z = plusMinus_one;

  return dirVector;
}

//______________________________________________________________________
// Compute the Ray direction from a cell face
__device__ void rayDirection_cellFaceDevice( curandState* randNumStates,
                                             const gpuIntVector& origin,
                                             const gpuIntVector& indexOrder, 
                                             const gpuIntVector& signOrder,
                                             const int iRay,
                                             gpuVector& directionVector,
                                             double& cosTheta)
{

  // Surface Way to generate a ray direction from the positive z face
  double phi   = 2 * M_PI * randDevice( randNumStates ); // azimuthal angle.  Range of 0 to 2pi
  double theta = acos( randDevice( randNumStates ) );      // polar angle for the hemisphere
  cosTheta = cos( theta );
  double sinTheta = sin( theta );

  //Convert to Cartesian
  gpuVector tmp;
  tmp[0] =  sinTheta * cos( phi );
  tmp[1] =  sinTheta * sin( phi );
  tmp[2] =  cosTheta;

  // Put direction vector as coming from correct face,
  directionVector[0] = tmp[indexOrder[0]] * signOrder[0];
  directionVector[1] = tmp[indexOrder[1]] * signOrder[1];
  directionVector[2] = tmp[indexOrder[2]] * signOrder[2];
}


//______________________________________________________________________
//
__device__ gpuVector rayLocationDevice( curandState* randNumStates,
                                      const gpuIntVector origin,
                                      const double DyDx, 
                                      const double DzDx,
                                      const bool useCCRays)
{
  gpuVector location;
  if( useCCRays == false ){
    location.x =   (double) origin.x +  randDevice( randNumStates ) ;
    location.y =   (double) origin.y +  randDevice( randNumStates ) * DyDx ;
    location.z =   (double) origin.z +  randDevice( randNumStates)  * DzDx ;
  }else{
    location.x =   origin.x +  0.5 ;
    location.y =   origin.y +  0.5 * DyDx ;
    location.z =   origin.z +  0.5 * DzDx ;
  }
  return location;
}

//______________________________________________________________________
//  Compute the Ray location from a cell face
__device__ void rayLocation_cellFaceDevice( curandState* randNumStates,
                                            const gpuIntVector& origin,
                                            const gpuIntVector &indexOrder, 
                                            const gpuIntVector &shift, 
                                            const double &DyDx, 
                                            const double &DzDx,
                                            gpuVector& location)
{
  gpuVector tmp;
  tmp[0] =  randDevice( randNumStates ) ;
  tmp[1] =  0;
  tmp[2] =  randDevice( randNumStates ) * DzDx ;
  
  // Put point on correct face
  location[0] = tmp[indexOrder[0]] + (double)shift[0];
  location[1] = tmp[indexOrder[1]] + (double)shift[1] * DyDx;
  location[2] = tmp[indexOrder[2]] + (double)shift[2] * DzDx;

  location[0] += (double) origin.x;
  location[1] += (double) origin.y;
  location[2] += (double) origin.z;
}

//______________________________________________________________________
//
__device__ bool has_a_boundaryDevice(const gpuIntVector &c, 
                                     const GPUGridVariable<int>& celltype, 
                                     BoundaryFaces &boundaryFaces){

  gpuIntVector adj = c;
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
                                   const gpuVector& inv_direction_vector){
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
__device__ void reflect(double& fs,
                         gpuIntVector& cur,
                         gpuIntVector& prevCell,
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

//______________________________________________________________________
template< class T >
__device__ void updateSumIDevice ( gpuVector& ray_direction,
                                   gpuVector& ray_location,
                                   const gpuIntVector& origin,
                                   const gpuVector& Dx,
                                   const GPUGridVariable< T >& sigmaT4OverPi,
                                   const GPUGridVariable< T >& abskg,
                                   const GPUGridVariable<int>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags)

{

 
  gpuIntVector cur = origin;
  gpuIntVector prevCell = cur;
  // Step and sign for ray marching
  int step[3];                                          // Gives +1 or -1 based on sign    
  bool sign[3];                                                                            
                                                                                           
  gpuVector inv_ray_direction = 1.0/ray_direction;
/*`==========TESTING==========*/
#if DEBUG == 1
  printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x, origin.y, origin.z,ray_direction.x, ray_direction.y, ray_direction.z, ray_location.x, ray_location.y, ray_location.z);
  printf("        inv_ray_dir [%g,%g,%g]\n", inv_ray_direction.x,inv_ray_direction.y,inv_ray_direction.z);
#endif
/*===========TESTING==========`*/  

  findStepSizeDevice(step, sign, inv_ray_direction);
  gpuVector D_DxRatio = make_double3(1, Dx.y/Dx.x, Dx.z/Dx.x );

  gpuVector tMax;         // (mixing bools, ints and doubles)
  tMax.x = (origin.x + sign[0]               - ray_location.x) * inv_ray_direction.x ; 
  tMax.y = (origin.y + sign[1] * D_DxRatio.y - ray_location.y) * inv_ray_direction.y ; 
  tMax.z = (origin.z + sign[2] * D_DxRatio.z - ray_location.z) * inv_ray_direction.z ; 

  //Length of t to traverse one cell
  gpuVector tDelta;
  tDelta   = Abs(inv_ray_direction) * D_DxRatio;                              

  //Initializes the following values for each ray
  bool in_domain     = true;
  double tMax_prev   = 0;
  double intensity   = 1.0;
  double fs          = 1.0;
  int nReflect       = 0;                 // Number of reflections                         
  double optical_thickness      = 0;                                                       
  double expOpticalThick_prev   = 1.0;                                                     


#ifdef RAY_SCATTER
  double scatCoeff = RT_flags.sigmaScat;          //[m^-1]  !! HACK !! This needs to come from data warehouse
  if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff;
  double curLength = 0;
#endif

  //+++++++Begin ray tracing+++++++++++++++++++
  //Threshold while loop
  while ( intensity > RT_flags.threshold ){

    DIR face = NONE;
    
    while (in_domain){

      prevCell = cur;
      double disMin = -9;          // Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      if ( tMax.x < tMax.y ){        // X < Y
        if ( tMax.x < tMax.z ){      // X < Z
          face = X;
        } else {
          face = Z;
        }
      } else {
        if( tMax.y < tMax.z ){       // Y < Z
          face = Y;
        } else {
          face = Z;
        }
      }
      
      //__________________________________
      //  update marching variables
      cur[face]  = cur[face] + step[face];
      disMin     = (tMax[face] - tMax_prev);
      tMax_prev  = tMax[face];
      tMax[face] = tMax[face] + tDelta[face];

      ray_location.x = ray_location.x + (disMin  * ray_direction.x);
      ray_location.y = ray_location.y + (disMin  * ray_direction.y);
      ray_location.z = ray_location.z + (disMin  * ray_direction.z);
      
/*`==========TESTING==========*/
#if DEBUG == 1
if(origin.x == 0 && origin.y == 0 && origin.z ==0){
    printf( "            cur [%d,%d,%d] prev [%d,%d,%d] ", cur.x, cur.y, cur.z, prevCell.x, prevCell.y, prevCell.z);
    printf( " face %d ", face ); 
    printf( "tMax [%g,%g,%g] ",tMax.x,tMax.y, tMax.z);
    printf( "rayLoc [%g,%g,%g] ",ray_location.x,ray_location.y, ray_location.z);
    printf( "inv_dir [%g,%g,%g] ",inv_ray_direction.x,inv_ray_direction.y, inv_ray_direction.z); 
    printf( "disMin %g \n",disMin ); 
   
    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevCell],  sigmaT4OverPi[prevCell]);
    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %i\n",abskg[cur], sigmaT4OverPi[cur], celltype[cur] );
} 
#endif

/*===========TESTING==========`*/
      in_domain = (celltype[cur]==-1);  //cellType of -1 is flow
    
      optical_thickness += Dx.x * abskg[prevCell]*disMin; // as long as tDeltaY,Z tMax.y(),Z and ray_location[1],[2]..
      // were adjusted by DyDx  or DzDx, this line is now correct for noncubic domains.

      RT_flags.nRaySteps ++;

      //Eqn 3-15(see below reference) while
      //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevCell] * ( expOpticalThick_prev - expOpticalThick ) * fs;

      expOpticalThick_prev = expOpticalThick;


#ifdef RAY_SCATTER
      curLength += disMin * Dx.x;
      if (curLength > scatLength && in_domain){

        // get new scatLength for each scattering event
        scatLength = -log( randDblExcDevice( randNumStates ) ) / scatCoeff; 

        ray_direction     = findRayDirectionDevice( randNumStates ); 
        
        inv_ray_direction = 1.0/ray_direction;

        // get new step and sign
        int stepOld = step[face];
        findStepSizeDevice( step, sign, inv_ray_direction);

        // if sign[face] changes sign, put ray back into prevCell (back scattering)
        // a sign change only occurs when the product of old and new is negative
        if( step[face] * stepOld < 0 ){
          cur = prevCell;
        }

        // get new tMax (mixing bools, ints and doubles)
        tMax.x = ( ( cur.x + sign[0]               - ray_location.x) * inv_ray_direction.x );
        tMax.y = ( ( cur.y + sign[1] * D_DxRatio.y - ray_location.y) * inv_ray_direction.y );
        tMax.z = ( ( cur.z + sign[2] * D_DxRatio.z - ray_location.z) * inv_ray_direction.z );

        // Length of t to traverse one cell
        tDelta    = Abs(inv_ray_direction) * D_DxRatio;
        tMax_prev = 0;
        curLength = 0;  // allow for multiple scattering events per ray

/*`==========TESTING==========*/
#if DEBUG == 3
        printf( "%i, %i, %i, tmax: %g, %g, %g  tDelta: %g, %g, %g \n", cur.x, cur.y, cur.z, tMax.x, tMax.y, tMax.z, tDelta.x, tDelta.y , tDelta.z );
#endif
/*===========TESTING==========`*/

        //if(_benchmark == 4 || _benchmark ==5) scatLength = 1e16; // only for Siegel Benchmark4 benchmark5. Only allows 1 scatter event.
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
#if DEBUG == 1
if(origin.x == 0 && origin.y == 0 && origin.z ==0 ){
    printf( "            cur [%d,%d,%d] intensity: %g expOptThick: %g, fs: %g allowReflect: %i \n", 
            cur.x, cur.y, cur.z, intensity,  exp(-optical_thickness), fs,RT_flags.allowReflect );
    
} 
__syncthreads();
#endif 
/*===========TESTING==========`*/
    //__________________________________
    //  Reflections
    if ( (intensity > RT_flags.threshold) && RT_flags.allowReflect){
      reflect( fs, cur, prevCell, abskg[cur], in_domain, step[face], sign[face], ray_direction[face]);
      ++nReflect;
    }

  }  // threshold while loop.
} // end of updateSumI function


//---------------------------------------------------------------------------
// Returns random number between 0 & 1.0 including 0 & 1.0
// See src/Core/Math/MersenneTwister.h for equation 
//---------------------------------------------------------------------------
__device__ double randDevice(curandState* globalState)
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

//---------------------------------------------------------------------------
// Returns random number between 0 & 1.0 excluding 0 & 1.0 
// See src/Core/Math/MersenneTwister.h for equation 
//---------------------------------------------------------------------------
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
//  Each thread gets same seed, a different sequence number, no offset 
//  This will create repeatable results.
__global__ void setupRandNum_kernel(curandState* randNumStates)
{
  int tID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  curand_init(1234, tID, 0, &randNumStates[tID]);
}

//______________________________________________________________________
template< class T>
__host__ void launchRayTraceKernel(dim3 dimGrid,
                                   dim3 dimBlock,
                                   int matlIndex,
                                   patchParams patch,
                                   cudaStream_t* stream,
                                   RMCRT_flags RT_flags,
                                   varLabelNames labelNames,
                                   GPUDataWarehouse* abskg_gdw,
                                   GPUDataWarehouse* sigmaT4_gdw,
                                   GPUDataWarehouse* celltype_gdw,
                                   GPUDataWarehouse* old_gdw,
                                   GPUDataWarehouse* new_gdw)
{
  // setup random number generator states on the device, 1 for each thread
  curandState* randNumStates;
  int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
  CUDA_RT_SAFE_CALL( cudaMalloc((void**)&randNumStates, numStates * sizeof(curandState)) );
  
  
  setupRandNum_kernel<<< dimGrid, dimBlock>>>( randNumStates );
  
  rayTraceKernel< T ><<< dimGrid, dimBlock, 0, *stream >>>(dimGrid, 
                                                      dimBlock, 
                                                      matlIndex,
                                                      patch,
                                                      randNumStates,
                                                      RT_flags,
                                                      labelNames,
                                                      abskg_gdw,
                                                      sigmaT4_gdw,
                                                      celltype_gdw,
                                                      old_gdw,
                                                      new_gdw);
    // free device-side RNG states
    CUDA_RT_SAFE_CALL( cudaFree(randNumStates) );
}

//______________________________________________________________________
//  Explicit template instantiations

template
__host__ void launchRayTraceKernel<double>(dim3 dimGrid,
                                           dim3 dimBlock,
                                           int matlIndex,
                                           patchParams patch,
                                           cudaStream_t* stream,
                                           RMCRT_flags RT_flags,
                                           varLabelNames labelNames,
                                           GPUDataWarehouse* abskg_gdw,
                                           GPUDataWarehouse* sigmaT4_gdw,
                                           GPUDataWarehouse* celltype_gdw,
                                           GPUDataWarehouse* old_gdw,
                                           GPUDataWarehouse* new_gdw);
                                   
template
__host__ void launchRayTraceKernel<float>(dim3 dimGrid,
                                          dim3 dimBlock,
                                          int matlIndex,
                                          patchParams patch,
                                          cudaStream_t* stream,
                                          RMCRT_flags RT_flags,
                                          varLabelNames labelNames,
                                          GPUDataWarehouse* abskg_gdw,
                                          GPUDataWarehouse* sigmaT4_gdw,
                                          GPUDataWarehouse* celltype_gdw,
                                          GPUDataWarehouse* old_gdw,
                                          GPUDataWarehouse* new_gdw);

} //end namespace Uintah
