/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <Core/Grid/Task.h>

#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"               // needed for max()
#undef __CUDA_INTERNAL_COMPILATION__

namespace Uintah {

typedef Uintah::gpuIntVector GPUIntVector;
typedef Uintah::gpuVector    GPUVector;
typedef Uintah::gpuPoint     GPUPoint;

//______________________________________________________________________
//
const int d_MAXLEVELS = 5;               // FIX ME!
const int d_MAX_RAYS = 500;
const int d_flowCell = -1;               // HARDWIRED!
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
//  returns const gpuIntVector - const gpuIntVector
inline HOST_DEVICE GPUIntVector operator-(const GPUIntVector& a, const GPUIntVector& b)
{
  return make_int3(a.x-b.x, a.y-b.y, a.z-b.z);
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

inline HOST_DEVICE bool operator!=(const GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x != b.x) || (a.y != b.y) || (a.z != b.z ));
}

inline HOST_DEVICE bool operator!=(GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x != b.x) || (a.y != b.y) || (a.z != b.z ));
}
inline HOST_DEVICE bool operator!=(const GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x != b.x) || (a.y != b.y) || (a.z != b.z ));
}

//__________________________________
//  returns gpuIntVector > gpuIntVector
inline HOST_DEVICE bool operator>(GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x > b.x) || (a.y > b.y) || (a.z > b.z ));
}

inline HOST_DEVICE bool operator>(const GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x > b.x) || (a.y > b.y) || (a.z > b.z ));
}

inline HOST_DEVICE bool operator>(GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x > b.x) || (a.y > b.y) || (a.z > b.z ));
}
inline HOST_DEVICE bool operator>(const GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x > b.x) || (a.y > b.y) || (a.z > b.z ));
}

//__________________________________
//  returns gpuIntVector < gpuIntVector
inline HOST_DEVICE bool operator<(GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x < b.x) || (a.y < b.y) || (a.z < b.z ));
}

inline HOST_DEVICE bool operator<(const GPUIntVector& a, GPUIntVector& b)
{
  return ( (a.x < b.x) || (a.y < b.y) || (a.z < b.z ));
}

inline HOST_DEVICE bool operator<(GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x < b.x) || (a.y < b.y) || (a.z < b.z ));
}
inline HOST_DEVICE bool operator<(const GPUIntVector& a, const GPUIntVector& b)
{
  return ( (a.x < b.x) || (a.y < b.y) || (a.z < b.z ));
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
class unifiedMemory {                     // this should be moved upstream
public:                                   // This only works for cuda > 6.X
#if 0       // turn off until titan has cuda > 6.0 installed
  void *operator new(size_t len)
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  void operator delete(void *ptr)
  {
    cudaFree(ptr);
  }

  void *operator new[] (size_t len)
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  void operator delete[] (void* ptr)
  {
    cudaFree(ptr);
  }
#endif
};
#if 0      // turn off until titan has cuda > 6.0 installed
//______________________________________________________________________
//
//  http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
// String Class for unified managed Memory
class GPUString : public unifiedMemory
{
  int length;
  char *data;

  public:
    GPUString() : length(0), data(0) {}
    // Constructor for C-GPUString initializer
    GPUString(const char *s) : length(0), data(0)
    {
      _realloc(strlen(s));
      strcpy(data, s);
    }

    // Copy constructor
    GPUString(const GPUString& s) : length(0), data(0)
    {
      _realloc(s.length);
      strcpy(data, s.data);
    }

    // destructor
    ~GPUString() {
      cudaFree(data);
    }

    // Assignment operator
    GPUString& operator=(const char* s)
    {
      _realloc(strlen(s));
      strcpy(data, s);
      return *this;
    }

    // Element access (from host or device)
    __host__ __device__
    char& operator[](int pos)
    {
      return data[pos];
    }

    // C-String access host or device
    __host__ __device__
    const char* c_str() const
    {
      return data;
    }

  private:
    void _realloc(int len)
    {
      cudaFree(data);
      length = len;
      cudaMallocManaged(&data, length+1);
    }
};
#endif
//______________________________________________________________________
//
struct varLabelNames : public unifiedMemory {
  public:

#if 0         // turn off until titan has cuda > 6.0 installed
    GPUString divQ;
    GPUString abskg;
    GPUString sigmaT4;
    GPUString celltype;
    GPUString VRFlux;
    GPUString boundFlux;
    GPUString radVolQ;

    __host__ __device__
    void print() {
      printf( " varLabelNames:  divQ: (%s), abskg: (%s), sigmaT4: (%s) ",divQ.c_str(), abskg.c_str(), sigmaT4.c_str() );
      printf( " celltype: (%s), VRFlux: (%s), boundFlux: (%s) \n",celltype.c_str(), VRFlux.c_str(), boundFlux.c_str() );
    }
#endif
};

//______________________________________________________________________
//
struct patchParams {
    GPUIntVector lo;          // cell low index not including extra or ghost cells
    GPUIntVector hi;          // cell high index not including extra or ghost cells
    GPUIntVector loEC;        // low index including extraCells
    GPUIntVector hiEC;        // high index including extraCells
    GPUVector    dx;          // cell spacing
    int          ID;          // patch ID
    //__________________________________
    //
    __host__ __device__
    void print() {
      printf( " patchParams: patchID: %i, lo[%i,%i,%i], hi: [%i,%i,%i])", ID, lo.x, lo.y, lo.z, hi.x,hi.y,hi.z);
      printf( " loEC: [%i,%i,%i], hiEC: [%i,%i,%i]\n  ",loEC.x, loEC.y, loEC.z, hiEC.x,hiEC.y,hiEC.z);
    }
};



//______________________________________________________________________
//
struct levelParams {
    GPUVector    Dx;                // cell spacing
    GPUIntVector regionLo;          // never use these regionLo/Hi in the kernel
    GPUIntVector regionHi;          // they vary on every patch and must be passed into the kernel
    bool         hasFinerLevel;
    int          index;             // level index
    GPUIntVector refinementRatio;
    GPUPoint     anchor;            // level anchor
    
    //__________________________________
    __host__ __device__ int iMax(int a, int b)
    {
      return a > b ? a : b;
    }
    
   //__________________________________
   //
    __host__ __device__
    int getCoarserLevelIndex() {
      int coarserLevel = iMax(index-1, 0 );
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
      printf( " LevelParams: hasFinerlevel: %i Dx: [%g,%g,%g] ",hasFinerLevel, Dx.x, Dx.y, Dx.z);
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
    bool usingFloats;               // if the communicated vars (sigmaT4 & abskg) are floats

    double sigma;                   // StefanBoltzmann constant
    double sigmaScat;               // scattering coefficient
    double threshold;

    int nDivQRays;                  // number of rays per cell used to compute divQ
    int nFluxRays;                  // number of boundary flux rays
    int nRaySteps;                  // number of ray steps taken
    int whichROI_algo;              // which Region of Interest algorithm
    int rayDirSampleAlgo;           // which ray direction sampling algorithm (Monte-Carlo or Latin-Hyper_Cube)
    unsigned int startCell {0};
    unsigned int endCell {0};
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

enum rayDirSampleAlgorithm{NAIVE, LATIN_HYPER_CUBE};
//
enum DIR {X=0, Y=1, Z=2, NONE=-9};
//           -x      +x       -y       +y     -z     +z
enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};
//
enum ROI_algo{  fixed,                // user specifies fixed low and high point for a bounding box 
                dynamic,              // user specifies thresholds that are used to dynamically determine ROI
                patch_based,          // The patch extents + halo are the ROI
                boundedRayLength,     // the patch extents + boundedRayLength/Dx are the ROI
                entireDomain          // The ROI is the entire computatonal Domain
             };
//

//______________________________________________________________________
//
__device__ GPUVector findRayDirectionDevice(curandState* randNumStates);

//______________________________________________________________________
//
__device__ GPUVector findRayDirectionHyperCubeDevice(curandState* randNumStates,
                                                     const int nDivQRays,
                                                     const int bin_i,
                                                     const int bin_j);
//______________________________________________________________________
//
__device__ void randVectorDevice( int int_array[],
                                  const int size,
                                  curandState* randNumStates );
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
__device__ void rayDirectionHyperCube_cellFaceDevice(curandState* randNumStates,
                                                     const GPUIntVector& origin,
                                                     const int3& indexOrder,
                                                     const int3& signOrder,
                                                     const int iRay,
                                                     GPUVector& dirVector,
                                                     double& cosTheta,
                                                     const int bin_i,
                                                     const int bin_j,
                                                     const int nFluxRays);

//______________________________________________________________________
// 
__device__ GPUVector rayOriginDevice( curandState* randNumStates,
                                      const GPUPoint  CC_pos,
                                      const GPUVector dx,
                                      const bool   useCCRays);
                                      
//______________________________________________________________________
//
__device__ void rayLocation_cellFaceDevice( curandState* randNumStates,
                                            const int face,
                                            const GPUVector Dx,
                                            const GPUPoint CC_pos,
                                            GPUVector& rayOrigin);

//______________________________________________________________________
//
__device__ bool has_a_boundaryDevice(const GPUIntVector& c,
                                     const GPUGridVariable<int>& celltype,
                                     BoundaryFaces& boundaryFaces);

//______________________________________________________________________
//
__device__ void raySignStepDevice(GPUVector& sign,
                                  int cellStep[],
                                  const GPUVector& inv_direction_vector);

//______________________________________________________________________
//
__device__ bool containsCellDevice( GPUIntVector fineLevel_ROI_Lo,
                                    GPUIntVector fineLevel_ROI_Hi,
                                    GPUIntVector cell,
                                    const int dir);

//______________________________________________________________________
//
__device__ void reflectDevice(double& fs,
                              GPUIntVector& cur,
                              GPUIntVector& prevCell,
                              const double abskg,
                              bool& in_domain,
                              int& step,
                              double& sign,
                              double& ray_direction);

//______________________________________________________________________
//
template<class T>
__device__ void updateSumIDevice ( levelParams level,
                                   GPUVector& ray_direction,
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
                                       const int3* regionLo,
                                       const int3* regionHi,
                                       const GPUGridVariable< T >*  sigmaT4OverPi,
                                       const GPUGridVariable< T >* abskg,
                                       const GPUGridVariable<int>* celltype,
                                       double& sumI,
                                       curandState* randNumStates,
                                       RMCRT_flags RT_flags);

//______________________________________________________________________
//
__device__ void setupRandNumsSeedAndSequences(curandState* randNumStates,
                                              int numStates,
                                              unsigned long long patchID,
                                              unsigned long long curTimestep);

//______________________________________________________________________
//
__device__ double randDblExcDevice(curandState* randNumStates);


//______________________________________________________________________
//
__device__ double randDblDevice(curandState* randNumStates);

//______________________________________________________________________
//
__device__ int randIntDevice(curandState* randNumStates,
                             const int B);

//______________________________________________________________________
//
__device__ bool isDbgCellDevice( GPUIntVector me);

//______________________________________________________________________
//
template< class T>
__device__ void GPUVariableSanityCK(const GPUGridVariable<T>& Q,
                                    const GPUIntVector Lo,
                                    const GPUIntVector Hi);

//______________________________________________________________________
//
template< class T >
__host__ void launchRayTraceKernel( DetailedTask* dtask,
                                    dim3 dimGrid,
                                    dim3 dimBlock,
                                    const int matlIndex,
                                    levelParams level,
                                    patchParams patch,
                                    cudaStream_t* stream,
                                    RMCRT_flags RT_flags,
                                    int curTimestep,
                                    GPUDataWarehouse* abskg_gdw,
                                    GPUDataWarehouse* sigmaT4_gdw,
                                    GPUDataWarehouse* celltype_gdw,
                                    GPUDataWarehouse* old_gdw,
                                    GPUDataWarehouse* new_gdw );

//______________________________________________________________________
//
template< class T >
__host__ void launchRayTraceDataOnionKernel( DetailedTask* dtask,
                                             dim3 dimGrid,
                                             dim3 dimBlock,
                                             int matlIndex,
                                             patchParams patchP,
                                             gridParams gridP,
                                             levelParams* levelP,
                                             GPUIntVector fineLevel_ROI_Lo,
                                             GPUIntVector fineLevel_ROI_Hi,
                                             cudaStream_t* stream,
                                             RMCRT_flags RT_flags,
                                             int curTimestep,
                                             GPUDataWarehouse* abskg_gdw,
                                             GPUDataWarehouse* sigmaT4_gdw,
                                             GPUDataWarehouse* celltype_gdw,
                                             GPUDataWarehouse* old_gdw,
                                             GPUDataWarehouse* new_gdw );

} // end namespace Uintah

#endif // end #ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_CUH
