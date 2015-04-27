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

#ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_CUH
#define CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_CUH

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <Core/Geometry/GPUVector.h>

#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"               // needed for max()
#undef __CUDA_INTERNAL_COMPILATION__ 

namespace Uintah {

typedef SCIRun::gpuIntVector GPUIntVector;
typedef SCIRun::gpuVector    GPUVector;
typedef SCIRun::gpuPoint     GPUPoint;

//______________________________________________________________________
//
const int d_MAXLEVELS = 3;               // FIX ME!

//__________________________________
//  returns gpuVector * scalar
inline HOST_DEVICE GPUVector operator*(const GPUVector & a, double b)
{
  return make_double3(a.x*b, a.y*b, a.z*b);
}

//__________________________________
//  returns gpuVector * scalar
inline HOST_DEVICE GPUVector operator*(double b, const GPUVector & a)
{
  return make_double3(a.x*b, a.y*b, a.z*b);
}

//__________________________________
//  returns gpuVector * gpuVector
inline HOST_DEVICE GPUVector operator*(const GPUVector& a, const GPUVector& b)
{
  return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//__________________________________
//  returns gpuVector * gpuIntVector
inline HOST_DEVICE GPUVector operator*(const GPUVector& a, const GPUIntVector& b)
{
  return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}
//__________________________________
//  returns a += b
inline HOST_DEVICE GPUVector operator+=(GPUVector& a, const GPUVector& b)
{
  return make_double3(a.x+=b.x, 
                      a.y+=b.y, 
                      a.z+=b.z);
}

//__________________________________
//  returns gpuVector/scalar
inline HOST_DEVICE GPUVector operator/(const GPUVector & a, double b)
{
  b = 1.0f / b;
  return a*b;
}

//__________________________________
//  returns scalar/gpuVector
inline HOST_DEVICE GPUVector operator/(double a, const GPUVector& b)
{
  return make_double3( a/b.x, a/b.y, a/b.z);
}

//__________________________________
//  returns abs
inline HOST_DEVICE GPUVector Abs(const GPUVector& v)
{
  double x = v.x < 0 ? -v.x:v.x;
  double y = v.y < 0 ? -v.y:v.y;
  double z = v.z < 0 ? -v.z:v.z;
  return make_double3(x,y,z);
}

//__________________________________
//  returns const gpuIntVector + const gpuIntVector
inline HOST_DEVICE GPUIntVector operator+(const GPUIntVector& a, const GPUIntVector& b)
{
  return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

//__________________________________
//  returns const gpuIntVector / const gpuIntVector
inline HOST_DEVICE GPUIntVector operator/(const GPUIntVector& a, const GPUIntVector& b)
{
  return make_int3(a.x/b.x, a.y/b.y, a.z/b.z);
}

//__________________________________
//  returns const gpuIntVector / gpuIntVector
inline HOST_DEVICE GPUIntVector operator/( const GPUIntVector& a, GPUIntVector& b)
{
  return make_int3(a.x/b.x, a.y/b.y, a.z/b.z);
}

//__________________________________
//  returns const gpuIntVector * const gpuIntVector
inline HOST_DEVICE GPUIntVector operator*(const GPUIntVector& a, const GPUIntVector& b)
{
  return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//__________________________________
//  returns gpuIntVector * gpuIntVector
inline HOST_DEVICE GPUIntVector operator*(GPUIntVector& a, const GPUIntVector& b)
{
  return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//__________________________________
//  returns gpuIntVector * gpuIntVector
inline HOST_DEVICE GPUIntVector operator*( GPUIntVector& a, GPUIntVector& b)
{
  return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//__________________________________
//  returns gpuIntVector + gpuIntVector
inline HOST_DEVICE GPUIntVector operator+(GPUIntVector& a, GPUIntVector& b)
{
  return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

//__________________________________
//  returns gpuIntVector == gpuIntVector
inline HOST_DEVICE bool operator==(GPUIntVector& a, GPUIntVector& b)
{
  return ( a.x == b.x && a.y == b.y && a.z == b.z );
}

//__________________________________
//  returns const gpuIntVector == gpuIntVector
inline HOST_DEVICE bool operator==(const GPUIntVector& a, GPUIntVector& b)
{
  return ( a.x == b.x && a.y == b.y && a.z == b.z );
}

//__________________________________
//  returns gpuIntVector == gpuIntVector
inline HOST_DEVICE bool operator!=(GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x != b.x) || (a.y != b.y) || (a.z != b.z ));
}

//__________________________________
//  returns gpuPoint + gpuVector
inline HOST_DEVICE GPUPoint operator+(GPUPoint& p, GPUVector& b)
{
  return make_double3(p.x+b.x, p.y+b.y, p.z+b.z);
}
//______________________________________________________________________
//
//______________________________________________________________________
//
struct varLabelNames {
    const char* divQ;
    const char* abskg;
    const char* sigmaT4;
    const char* celltype;
    const char* VRFlux;
    const char* boundFlux;
    const char* radVolQ;
};

//______________________________________________________________________
//
struct patchParams {
    GPUIntVector lo;          // cell low index not including extra or ghost cells
    GPUIntVector hi;          // cell high index not including extra or ghost cells
    GPUIntVector loEC;        // low index including extraCells
    GPUIntVector hiEC;        // high index including extraCells
    GPUIntVector nCells;      // number of cells in each dir
    GPUVector    dx;          // cell spacing
    int          ID;          // patch ID
};



//______________________________________________________________________
//
struct levelParams {
    double       DyDx;
    double       DzDx;
    GPUVector    Dx;                // cell spacing
    GPUIntVector regionLo;
    GPUIntVector regionHi;
    bool         hasFinerLevel;
    int          index;             // level index
    GPUIntVector refinementRatio;
    GPUPoint     anchor;            // level anchor

   //__________________________________
   //
    __host__ __device__ 
    int getCoarserLevelIndex() {
      int coarserLevel = max(index-1, 0 );
      return coarserLevel; 
    } 

    //__________________________________
    //  GPU version of level::getCellPosition()
    __device__
    GPUPoint getCellPosition(const GPUIntVector& cell)
    { 
      double x = anchor.x + (Dx.x * cell.x) + (0.5 * Dx.x);
      double y = anchor.y + (Dx.y * cell.y) + (0.5 * Dx.y);
      double z = anchor.z + (Dx.z * cell.z) + (0.5 * Dx.z);
      return make_double3(x,y,z);
    }
    
    
    //__________________________________
    //  GPU version of level::mapCellToCoarser()
    __device__
    GPUIntVector mapCellToCoarser(const GPUIntVector& idx)
    { 
      GPUIntVector ratio = idx/refinementRatio;

      // If the fine cell index is negative
      // you must add an offset to get the right
      // coarse cell. -Todd
      GPUIntVector offset = make_int3(0,0,0);
     
      if ( (idx.x < 0) && (refinementRatio.x  > 1 )){
        offset.x = (int)fmod( (double)idx.x, (double)refinementRatio.x ) ;
      }
      if ( (idx.y < 0) && (refinementRatio.y > 1 )){
        offset.y = (int)fmod( (double)idx.y, (double)refinementRatio.y );
      }  

      if ( (idx.z < 0) && (refinementRatio.z > 1)){
        offset.z = (int) fmod((double)idx.z, (double)refinementRatio.z );
      }
      return ratio + offset;
    }
   
    //__________________________________
    //
    __host__ __device__ 
    void print() {
      printf( " LevelParams: hasFinerlevel: %i DyDz: %g  DzDz: %g, Dx: [%g,%g,%g] ",hasFinerLevel,DyDx,DzDx, Dx.x,Dx.y, Dx.z);
      printf( " regionLo: [%i,%i,%i], regionHi: [%i,%i,%i]\n  ",regionLo.x, regionLo.y, regionLo.z, regionHi.x, regionHi.y, regionHi.z);
      printf( " RefineRatio: [%i,%i,%i] ",refinementRatio.x, refinementRatio.y, refinementRatio.z);
    }
};

//______________________________________________________________________
//
struct BoundingBox {
    GPUPoint lo;
    GPUPoint hi;

    __device__
    bool inside(GPUPoint p)
    {
      return ( (p.x >= lo.x) && 
               (p.y >= lo.y) && 
               (p.z >= lo.z) && 
               (p.x <= hi.x) && 
               (p.y <= hi.y) && 
               (p.z <= hi.z) );
    }
};

//______________________________________________________________________
//
struct gridParams {
    int                maxLevels;
    struct BoundingBox domain_BB;
};

//______________________________________________________________________
//
struct RMCRT_flags {
    bool modifies_divQ;
    bool solveDivQ;
    bool allowReflect;
    bool solveBoundaryFlux;
    bool CCRays;
    bool usingFloats;           // if the communicated vars (sigmaT4 & abskg) are floats

    double sigma;               // StefanBoltzmann constant
    double sigmaScat;           // scattering coefficient
    double threshold;

    int nDivQRays;            // number of rays per cell used to compute divQ
    int nFluxRays;            // number of boundary flux rays
    int nRaySteps;            // number of ray steps taken
    int whichROI_algo;        // which Region of Interest algorithm
};

//__________________________________
//  Struct for managing the boundary faces
struct BoundaryFaces {
    __device__ BoundaryFaces()
        : nFaces(0)
    {
    }

    int nFaces;             // number of faces
    int faceArray[6];       // vector of faces

    // add Face to array
    __device__ void addFace(int f)
    {
      faceArray[nFaces] = f;
      nFaces++;
    }

    // returns the number of faces
    __device__ int size()
    {
      return nFaces;
    }

    // print facesArray
    __device__ void print(int tid)
    {
      for (int f = 0; f < nFaces; f++) {
        printf("  tid: %i face[%i]: %i\n", tid, f, faceArray[f]);
      }
    }

};


enum DIR {X=0, Y=1, Z=2, NONE=-9};
//           -x      +x       -y       +y     -z     +z
enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};
//
enum ROI_algo{fixed, dynamic, patch_based};

//______________________________________________________________________
//
__device__ GPUVector findRayDirectionDevice(curandState* randNumStates);

//______________________________________________________________________
//
__device__ void rayDirection_cellFaceDevice(curandState* randNumStates,
                                            const GPUIntVector& origin,
                                            const GPUIntVector& indexOrder,
                                            const GPUIntVector& signOrder,
                                            const int iRay,
                                            GPUVector& directionVector,
                                            double& cosTheta);
                            
//______________________________________________________________________
//
__device__ GPUVector rayLocationDevice(curandState* randNumStates,
                                       const GPUIntVector origin,
                                       const double DyDx,
                                       const double DzDx,
                                       const bool useCCRays);
                                      

__device__ void rayLocation_cellFaceDevice(curandState* randNumStates,
                                           const GPUIntVector& origin,
                                           const GPUIntVector& indexOrder,
                                           const GPUIntVector& shift,
                                           const double& DyDx,
                                           const double& DzDx,
                                           GPUVector& location);

//______________________________________________________________________
//
__device__ bool has_a_boundaryDevice(const GPUIntVector& c,
                                     const GPUGridVariable<int>& celltype,
                                     BoundaryFaces& boundaryFaces);

//______________________________________________________________________
//
__device__ void findStepSizeDevice(int step[],
                                   bool sign[],
                                   const GPUVector& inv_direction_vector);
 
//______________________________________________________________________
//
__device__ bool containsCellDevice( GPUIntVector fineLevel_ROI_Lo, 
                                    GPUIntVector fineLevel_ROI_Hi, 
                                    GPUIntVector cell, 
                                    const int dir); 
                                 
//______________________________________________________________________
//
__device__ void reflect(double& fs,
                        GPUIntVector& cur,
                        GPUIntVector& prevCell,
                        const double abskg,
                        bool& in_domain,
                        int& step,
                        bool& sign,
                        double& ray_direction);

//______________________________________________________________________
//
template<class T>                                                          
__device__ void updateSumIDevice ( GPUVector& ray_direction,
                                   GPUVector& ray_location,
                                   const GPUIntVector& origin,
                                   const GPUVector& Dx,
                                   const GPUGridVariable< T >&  sigmaT4OverPi,
                                   const GPUGridVariable< T >& abskg,
                                   const GPUGridVariable<int>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags);
                                   
//______________________________________________________________________
//  Multi-level

template< class T>
 __device__ void updateSumI_MLDevice ( GPUVector& ray_direction,
                                       GPUVector& ray_location,
                                       const GPUIntVector& origin,
                                       gridParams gridP,
                                       const GPUIntVector& fineLevel_ROI_Lo,
                                       const GPUIntVector& fineLevel_ROI_Hi,
                                       const GPUGridVariable< T >*  sigmaT4OverPi,
                                       const GPUGridVariable< T >* abskg,
                                       const GPUGridVariable<int>* celltype,
                                       double& sumI,
                                       curandState* randNumStates,
                                       RMCRT_flags RT_flags);

//______________________________________________________________________
//
__device__ double randDblExcDevice(curandState* randNumStates);


//______________________________________________________________________
//
__device__ double randDevice(curandState* randNumStates);

//______________________________________________________________________
//
__device__ bool isDbgCellDevice( GPUIntVector me);



//______________________________________________________________________
//
template< class T >
__host__ void launchRayTraceKernel( dim3 dimGrid,
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
                                    GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template< class T >
__global__ void rayTraceKernel( dim3 dimGrid,
                                dim3 dimBlock,
                                int matlIndex,
                                patchParams patch,
                                curandState* randNumStates,
                                RMCRT_flags RT_flags,
                                varLabelNames labelNames,
                                GPUDataWarehouse* abskg_gdw,
                                GPUDataWarehouse* sigmaT4_gdw,
                                GPUDataWarehouse* celltype_gdw,
                                GPUDataWarehouse* old_gdw,
                                GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template< class T >
__host__ void launchRayTraceDataOnionKernel( dim3 dimGrid,
                                             dim3 dimBlock,
                                             int matlIndex,
                                             patchParams patchP,
                                             gridParams gridP,
                                             levelParams* levelP,
                                             GPUIntVector fineLevel_ROI_Lo,
                                             GPUIntVector fineLevel_ROI_Hi,
                                             cudaStream_t* stream,
                                             RMCRT_flags RT_flags,
                                             GPUDataWarehouse* abskg_gdw,
                                             GPUDataWarehouse* sigmaT4_gdw,
                                             GPUDataWarehouse* celltype_gdw,
                                             GPUDataWarehouse* old_gdw,
                                             GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template< class T >
__global__ void rayTraceDataOnionKernel( dim3 dimGrid,
                                         dim3 dimBlock,
                                         int matl,
                                         patchParams finePatch,
                                         gridParams gridP,
                                         GPUIntVector fineLevel_ROI_Lo,
                                         GPUIntVector fineLevel_ROI_Hi,
                                         curandState* randNumStates,
                                         RMCRT_flags RT_flags,
                                         GPUDataWarehouse* abskg_gdw,
                                         GPUDataWarehouse* sigmaT4_gdw,
                                         GPUDataWarehouse* celltype_gdw,
                                         GPUDataWarehouse* old_gdw,
                                         GPUDataWarehouse* new_gdw );

} // end namespace Uintah

#endif // end #ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_CUH
