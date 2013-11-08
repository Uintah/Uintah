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

#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <Core/Grid/Variables/GPUGridVariable.h>

// linker support for device code not ready yet, need to include the whole source...
#include <CCA/Components/Schedulers/GPUDataWarehouse.cu>
#include <Core/Grid/Variables/Stencil7.h>
#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>


#define DEBUG


//__________________________________
//  To Do
//  - fix seed in random number generator
//  - Figure out how to initialize variables.
//  - add BoundaryFlux and cellType variables
//  - 


namespace Uintah {

//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel(dim3 dimGrid,
                               dim3 dimBlock,
                               int matl,
                               patchParams patch,
                               const Int3 domainLo,
                               const Int3 domainHi,
                               curandState* randNumStates,
                               RMCRT_flags RT_flags,
                               varLabelNames labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw)
{

  int blockID = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
  int threadID = threadIdx.x +  blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;
  
  if(blockID == 0 && threadID == 0){
    printf("//________________________________rayTraceKernek Modifies: %d\n", RT_flags.modifies_divQ);
    printf("  Top:RayTraceKernel %d,%d\n", threadIdx.x, threadIdx.y );
  }
  
  const GPUGridVariable<double> sigmaT4OverPi;
  const GPUGridVariable<double> abskg;              // Need to use getRegion() to get the data
  const GPUGridVariable<double> celltype;

  GPUGridVariable<double> divQ;
  GPUGridVariable<double> VRFlux;
  GPUGridVariable<Stencil7> boundFlux;
  GPUGridVariable<double> radiationVolQ;
 
  abskg_gdw->get(   abskg   ,       "abskg" ,   patch.ID, matl );  
  sigmaT4_gdw->get( sigmaT4OverPi , "sigmaT4",  patch.ID, matl );
  celltype_gdw->get( celltype ,     "DcellType", patch.ID, matl );

  if( RT_flags.modifies_divQ ){
    new_gdw->getModifiable( divQ,         "divQ",          patch.ID, matl );
    new_gdw->getModifiable( VRFlux,       "VRFlux",        patch.ID, matl );
/*`    new_gdw->getModifiable( boundFlux,    "boundFlux",     patch.ID, matl );      TESTING`*/
    new_gdw->getModifiable( radiationVolQ,"radiationVolq", patch.ID, matl );
  }else{
    new_gdw->get( divQ,         "divQ",          patch.ID, matl );         // these should be allocateAntPut() calls
    new_gdw->get( VRFlux,       "VRFlux",        patch.ID, matl );
/*`    new_gdw->get( boundFlux,    "boundFlux",     patch.ID, matl );      TESTING`*/
    new_gdw->get( radiationVolQ,"radiationVolq", patch.ID, matl );
    
 #if 0
       // Not sure how to initialize variables on GPU
    divQ.initialize( 0.0 ); 
    VRFlux.initialize( 0.0 );
    radiationVolq.initialize( 0.0 );

    for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector origin = *iter;

      boundFlux[origin].p = 0.0;
      boundFlux[origin].w = 0.0;
      boundFlux[origin].e = 0.0;
      boundFlux[origin].s = 0.0;
      boundFlux[origin].n = 0.0;
      boundFlux[origin].b = 0.0;
      boundFlux[origin].t = 0.0;
    }
#endif
  }

  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x + domainLo.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y + domainLo.y;

  // Get the extents of the data block in which the variables reside.
  // This is essentially the stride in the index calculations.
  Int3 nCells = patch.nCells;
  
  double DyDx = patch.dx.y/patch.dx.x;
  double DzDx = patch.dx.z/patch.dx.x;
  
  //______________________________________________________________________
  //           R A D I O M E T E R
  //______________________________________________________________________
  if ( RT_flags.virtRad ){  
  }
  
  //______________________________________________________________________
  //          B O U N D A R Y F L U X
  //______________________________________________________________________
  if( RT_flags.solveBoundaryFlux ){
  }
  
  
  //______________________________________________________________________
  //         S O L V E   D I V Q
  //______________________________________________________________________
  if( RT_flags.solveDivQ ){
    // GPU equivalent of GridIterator loop - calculate sets of rays per thread
    if (tidX >= patch.lo.x && tidY >= patch.lo.y && tidX < patch.hi.x && tidY < patch.hi.y) { // patch boundary check
      #pragma unroll
      for (int z = patch.lo.z; z < patch.hi.z; z++) { // loop through z slices
      
        // calculate the index for individual threads
        int idx = INDEX3D( nCells.x, nCells.y, tidX, tidY,z );

        Int3 origin = make_int3(tidX, tidY, z);  // for each thread
        double sumI = 0;
        
        //__________________________________
        // ray loop
        #pragma unroll
        for (int iRay = 0; iRay < RT_flags.nDivQRays; iRay++) {
        
          Double3 direction_vector = findRayDirectionDevice( randNumStates, RT_flags.isSeedRandom, origin, iRay, tidX );
          
          Double3 ray_location = rayLocationDevice( randNumStates, origin, DyDx,  DzDx, RT_flags.CCRays );
         
          updateSumIDevice( direction_vector, ray_location, origin, patch.dx,  sigmaT4OverPi, abskg, celltype, sumI, randNumStates, RT_flags);
        } //Ray loop
 
        //__________________________________
        //  Compute divQ
        divQ[origin] = 4.0 * M_PI * abskg[origin] * ( sigmaT4OverPi[origin] - (sumI/RT_flags.nDivQRays) );

        // radiationVolq is the incident energy per cell (W/m^3) and is necessary when particle heat transfer models (i.e. Shaddix) are used 
        radiationVolQ[origin] = 4.0 * M_PI * abskg[origin] *  (sumI/RT_flags.nDivQRays) ;
        
/*`==========TESTING==========*/
#ifdef DEBUG
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
__device__ Double3 findRayDirectionDevice(curandState* randNumStates,
                                          const bool isSeedRandom,
                                          const Int3 origin,
                                          const int iRay,
                                          const int tidX)
{
#if 0
  if( isSeedRandom == false ){
   // mTwister.seed((origin.x() + origin.y() + origin.z()) * iRay +1);
   curand_init( hashDevice(tidX), tidX, 0, &randNumStates[tidX] );        // TODD FIX THIS
  }

  // Random Points On Sphere
  double plusMinus_one = 2 * randDblExcDevice( randNumStates ) - 1;
  double r = sqrt(1 - plusMinus_one * plusMinus_one);             // Radius of circle at z
  double theta = 2 * M_PI * randDblExcDevice( randNumStates );    // Uniform betwen 0-2Pi

  Double3 dirVector;
  dirVector.x = r*cos(theta);                     // Convert to cartesian
  dirVector.y = r*sin(theta);
  dirVector.z = plusMinus_one;

#endif  
/*`==========TESTING==========*/
  Double3 dirVector;
  dirVector.x = 1.;
  dirVector.y = 1.;
  dirVector.z = 1.; 
/*===========TESTING==========`*/
  
  
  return dirVector;
}

//______________________________________________________________________
//
__device__ Double3 rayLocationDevice( curandState* randNumStates,
                                      const Int3 origin,
                                      const double DyDx, 
                                      const double DzDx,
                                      const bool useCCRays)
{
  Double3 location;
  if( useCCRays == false ){
    location.x =   origin.x +  randDevice( randNumStates ) ;
    location.y =   origin.y +  randDevice( randNumStates ) * DyDx ;
    location.z =   origin.z +  randDevice( randNumStates)  * DzDx ;
  }else{
    location.x =   origin.x +  0.5 ;
    location.y =   origin.y +  0.5 * DyDx ;
    location.z =   origin.z +  0.5 * DzDx ;
  }
  return location;
}


//______________________________________________________________________
//
__device__ void findStepSizeDevice(int step[],
                                   bool sign[],
                                   const Double3& inv_direction_vector){
  // get new step and sign
  for ( int d= 0; d<3; d++){
  
    if (inv_direction_vector[d]>0){
      step[d] = 1;
      sign[d] = 1;
    }
    else{
      step[d] = -1;
      sign[d] = 0;
    }
  }
}


//______________________________________________________________________
//
__device__ void reflect(double& fs,
                         Int3& cur,
                         Int3& prevCell,
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
__device__ void updateSumIDevice ( Double3& ray_direction,
                                   Double3& ray_location,
                                   const Int3& origin,
                                   const Double3& Dx,
                                   const GPUGridVariable<double>& sigmaT4OverPi,
                                   const GPUGridVariable<double>& abskg,
                                   const GPUGridVariable<double>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags)

{

/*`==========TESTING==========*/
  printf("        updateSumI: [%d,%d,%d] ray_dir [%g,%g,%g] ray_loc [%g,%g,%g]\n", origin.x, origin.y, origin.z,ray_direction.x, ray_direction.y, ray_direction.z, ray_location.x, ray_location.y, ray_location.z);
/*===========TESTING==========`*/  
  Int3 cur = origin;
  Int3 prevCell = cur;
  // Step and sign for ray marching
  int step[3];                                          // Gives +1 or -1 based on sign    
  bool sign[3];                                                                            
                                                                                           
  Double3 inv_ray_direction = 1.0/ray_direction;


  findStepSizeDevice(step, sign, inv_ray_direction);
  Double3 D_DxRatio = make_double3(1, Dx.y/Dx.x, Dx.z/Dx.x );

  Double3 tMax;         // (mixing bools, ints and doubles)
  tMax.x = (origin.x + sign[0]               - ray_location.x) * inv_ray_direction.x ; 
  tMax.y = (origin.y + sign[1] * D_DxRatio.y - ray_location.y) * inv_ray_direction.y ; 
  tMax.z = (origin.z + sign[2] * D_DxRatio.z - ray_location.z) * inv_ray_direction.z ; 

  //Length of t to traverse one cell
  Double3 tDelta; 
  tDelta.x = abs( inv_ray_direction.x );
  tDelta.y = abs( inv_ray_direction.y ) * D_DxRatio.y;
  tDelta.z = abs( inv_ray_direction.z ) * D_DxRatio.z;                                      

  //Initializes the following values for each ray
  bool in_domain     = true;
  double tMax_prev   = 0;
  double intensity   = 1.0;
  double fs          = 1.0;
  int nReflect       = 0;                 // Number of reflections                         
  double optical_thickness      = 0;                                                       
  double expOpticalThick_prev   = 1.0;                                                     


#ifdef RAY_SCATTER
  double scatCoeff = _sigmaScat;          //[m^-1]  !! HACK !! This needs to come from data warehouse
  if (scatCoeff == 0) scatCoeff = 1e-99;  // avoid division by zero

  // Determine the length at which scattering will occur
  // See CCA/Components/Arches/RMCRT/PaulasAttic/MCRT/ArchesRMCRT/ray.cc
  double scatLength = -log( randDblExc(randNumStates) ) / scatCoeff;
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
#if 1
if(origin.x == 0 && origin.y == 0 && origin.z ==0){
    printf( "            cur [%d,%d,%d] prev [%d,%d,%d] ", cur.x, cur.y, cur.z, prevCell.x, prevCell.y, prevCell.z);
    printf( " face %d ", face ); 
    printf( "tMax [%g,%g,%g] ",tMax.x,tMax.y, tMax.z);
    printf( "rayLoc [%g,%g,%g] ",ray_location.x,ray_location.y, ray_location.z);
    printf( "inv_dir [%g,%g,%g] ",inv_ray_direction.x,inv_ray_direction.y, inv_ray_direction.z); 
    printf( "disMin %g \n",disMin ); 
   
    printf( "            abskg[prev] %g  \t sigmaT4OverPi[prev]: %g \n",abskg[prevCell],  sigmaT4OverPi[prevCell]);
    printf( "            abskg[cur]  %g  \t sigmaT4OverPi[cur]:  %g  \t  cellType: %g\n",abskg[cur], sigmaT4OverPi[cur], celltype[cur] );
} 
#endif

/*===========TESTING==========`*/
      //in_domain = (celltype[cur]==-1);  //cellType of -1 is flow

      if( (celltype[cur] + 1 < 10 * DBL_EPSILON) ) {
        in_domain = true;
      }else{
        in_domain = false;
      } 
    
      optical_thickness += Dx.x * abskg[prevCell]*disMin; // as long as tDeltaY,Z tMax.y(),Z and ray_location[1],[2]..
      // were adjusted by DyDx  or DzDx, this line is now correct for noncubic domains.

      RT_flags.nRaySteps ++;

      //Eqn 3-15(see below reference) while
      //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
      double expOpticalThick = exp(-optical_thickness);

      sumI += sigmaT4OverPi[prevCell] * ( expOpticalThick_prev - expOpticalThick ) * fs;

      expOpticalThick_prev = expOpticalThick;

#ifdef RAY_SCATTER
      curLength += disMin * Dx.x(); // July 18
      if (curLength > scatLength && in_domain){

        // get new scatLength for each scattering event
        scatLength = -log(mTwister.randDblExc() ) / scatCoeff; 

        ray_direction     =  findRayDirection( mTwister, _isSeedRandom, cur ); 
        inv_ray_direction = Vector(1.0)/ray_direction;

        // get new step and sign
        int stepOld = step[face];
        findStepSize( step, sign, inv_ray_direction);

        // if sign[face] changes sign, put ray back into prevCell (back scattering)
        // a sign change only occurs when the product of old and new is negative
        if( step[face] * stepOld < 0 ){
          cur = prevCell;
        }

        // get new tMax (mixing bools, ints and doubles)
        tMax.x = ( ( cur.x + sign.x               - ray_location.x) * inv_ray_direction.x );
        tMax.y = ( ( cur.y + sign.y * D_DxRatio.y - ray_location.y) * inv_ray_direction.y );
        tMax.z = ( ( cur.z + sign.z * D_DxRatio.z - ray_location.z) * inv_ray_direction.z );

        // Length of t to traverse one cell
        tDelta    = Abs(inv_ray_direction) * D_DxRatio;
        tMax_prev = 0;
        curLength = 0;  // allow for multiple scattering events per ray

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
#if 1
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
// Device Function:
//---------------------------------------------------------------------------
__device__ bool containsCellDevice(const Int3& domainLo,
                                   const Int3& domainHi,
                                   const Int3& cell,
                                   const int& face)
{
  switch (face) {
    case 0 :
      return domainLo.x <= cell.x && domainHi.x > cell.x;
    case 1 :
      return domainLo.y <= cell.y && domainHi.y > cell.y;
    case 2 :
      return domainLo.z <= cell.z && domainHi.z > cell.z;
    default :
      return false;
  }
}

//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ double randDevice(curandState* globalState)
{
#if 0
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    double val = curand(&localState);
    globalState[tid] = localState;
#endif
    
/*`==========TESTING==========*/
    return 0.5;        
    //return (double)val * (1.0/4294967295.0); 
/*===========TESTING==========`*/
}

//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ double randDblExcDevice(curandState* globalState)
{
#if 0
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    double val = curand_uniform(&localState);
    globalState[tid] = localState;
#endif

/*`==========TESTING==========*/
    return 0.5;
    //return ( (double)val + 0.5 ) * (1.0/4294967296.0); 
/*===========TESTING==========`*/
}

//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ unsigned int hashDevice(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);

    return a;
}

//______________________________________________________________________

__host__ void launchRayTraceKernel(dim3 dimGrid,
                                   dim3 dimBlock,
                                   int matlIndex,
                                   patchParams patch,
                                   const Int3 domainLo,
                                   const Int3 domainHi,
                                   curandState* globalDevRandStates,
                                   cudaStream_t* stream,
                                   RMCRT_flags RT_flags,
                                   varLabelNames labelNames,
                                   GPUDataWarehouse* abskg_gdw,
                                   GPUDataWarehouse* sigmaT4_gdw,
                                   GPUDataWarehouse* celltype_gdw,
                                   GPUDataWarehouse* old_gdw,
                                   GPUDataWarehouse* new_gdw)
{

  rayTraceKernel<<< dimGrid, dimBlock, 0, *stream >>>(dimGrid, 
                                                      dimBlock, 
                                                      matlIndex,
                                                      patch,
                                                      domainLo, 
                                                      domainHi,
                                                      globalDevRandStates,
                                                      RT_flags,
                                                      labelNames,
                                                      abskg_gdw,
                                                      sigmaT4_gdw,
                                                      celltype_gdw,
                                                      old_gdw,
                                                      new_gdw);
}

} //end namespace Uintah
