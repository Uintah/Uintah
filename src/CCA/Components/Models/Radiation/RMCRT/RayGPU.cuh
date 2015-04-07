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

namespace Uintah {

typedef SCIRun::gpuIntVector GPUIntVector;
typedef SCIRun::gpuVector    GPUVector;
typedef SCIRun::gpuPoint     GPUPoint;

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
    GPUVector    Dx;
    GPUIntVector regionLo;
    GPUIntVector regionHi;
    bool         hasFinerLevel;
    
    __host__ __device__ 
    void print() {
      printf( " LevelParams: hasFinerlevel: %i DyDz: %g  DzDz: %g, Dx: [%g,%g,%g] ",hasFinerLevel,DyDx,DzDx, Dx.x,Dx.y, Dx.z);
      printf( " regionLo: [%i,%i,%i], regionHi: [%i,%i,%i]\n  ",regionLo.x, regionLo.y, regionLo.z, regionHi.x, regionHi.y, regionHi.z);
    }
};

//______________________________________________________________________
//
struct BoundingBox {
    GPUPoint lo;
    GPUPoint hi;
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
//
__device__ double randDblExcDevice(curandState* randNumStates);


//______________________________________________________________________
//
__device__ double randDevice(curandState* randNumStates);


//______________________________________________________________________
//
__device__ unsigned int hashDevice(unsigned int a);


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
                                             //__________________________________
                                             //  FIX ME
                                             GPUVector Dx_0, GPUVector Dx_1, 
                                             bool hasFinerLevel_0, bool hasFinerLevel_1,
                                             double DyDx_0, double DyDx_1,
                                             double DzDx_0, double DzDx_1,
                                             GPUVector regionLo_0, GPUVector regionLo_1,
                                             GPUVector regionHi_0, GPUVector regionHi_1,
                                             GPUIntVector fineLevel_ROI_Lo, GPUIntVector fineLevel_ROI_Hi,
                                             //__________________________________
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
                                         patchParams patch,
                                         gridParams gridP,
                                         levelParams* levelP,  // array of levelParam structs
                                        //__________________________________
                                        //  fix ME!!!
                                         GPUVector Dx_0, GPUVector Dx_1,
                                         bool hasFinerLevel_0, bool hasFinerLevel_1,
                                         double DyDx_0, double DyDx_1,
                                         double DzDx_0, double DzDx_1,
                                         GPUVector regionLo_0, GPUVector regionLo_1,
                                         GPUVector regionHi_0, GPUVector regionHi_1,
                                         GPUIntVector fineLevel_ROI_Lo, GPUVector fineLevel_ROI_Hi,
                                         //__________________________________
                                         curandState* randNumStates,
                                         RMCRT_flags RT_flags,
                                         GPUDataWarehouse* abskg_gdw,
                                         GPUDataWarehouse* sigmaT4_gdw,
                                         GPUDataWarehouse* celltype_gdw,
                                         GPUDataWarehouse* old_gdw,
                                         GPUDataWarehouse* new_gdw );

} // end namespace Uintah

#endif // end #ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYGPU_CUH
