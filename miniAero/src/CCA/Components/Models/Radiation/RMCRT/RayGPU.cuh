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

#ifndef RAY_GPU_CUH
#define RAY_GPU_CUH

#include <Core/Geometry/GPUVector.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>

namespace Uintah {

using namespace std;
using namespace SCIRun;

struct varLabelNames{
  const char* divQ;
  const char* abskg;
  const char* sigmaT4;
  const char* celltype;
  const char* VRFlux;
  const char* boundFlux;
  const char* radVolQ;
};


struct patchParams{
  gpuVector dx;             // cell spacing
  gpuIntVector lo;          // cell low index not including extra or ghost cells
  gpuIntVector hi;          // cell high index not including extra or ghost cells
  gpuIntVector loEC;        // low index including extraCells
  gpuIntVector hiEC;        // high index including extraCells
  gpuIntVector nCells;      // number of cells in each dir
  int ID;                   // patch ID
};

struct RMCRT_flags{
  bool modifies_divQ;
  bool solveDivQ;
  bool allowReflect;
  bool solveBoundaryFlux;
  bool CCRays;
  bool usingFloats;           // if the communicated vars (sigmaT4 & abskg) are floats
  
  double sigma;               // StefanBoltzmann constant
  double sigmaScat;           // scattering coefficient
  double threshold;
  
  int   nDivQRays;            // number of rays per cell used to compute divQ
  int   nFluxRays;            // number of boundary flux rays
  int   nRaySteps;            // number of ray steps taken
};

//__________________________________
//  Struct for managing the boundary faces
struct BoundaryFaces{
  __device__ BoundaryFaces():nFaces(0){}
  
  int nFaces;             // number of faces
  int faceArray[6];       // vector of faces
  
  // add Face to array
  __device__ void addFace(int f){
    faceArray[nFaces] = f;
    nFaces ++;    
  }
 
  
  // returns the number of faces
  __device__ int size(){
    return nFaces;
  }
  
  // print facesArray
  __device__ void print(int tid){
    for(int f=0; f<nFaces; f++){
      printf("  tid: %i face[%i]: %i\n",tid,f, faceArray[f]);
    }
  }
  
};


enum DIR {X=0, Y=1, Z=2, NONE=-9};
//           -x      +x       -y       +y     -z     +z
enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};

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
                                   GPUDataWarehouse* new_gdw);


template< class T>
__global__ void rayTraceKernel(dim3 dimGrid,
                               dim3 dimBlock,
                               int matlIndex,
                               patchParams patch,
                               curandState randNumStates,
                               RMCRT_flags RT_flags,
                               varLabelNames* labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw);
                               
__device__ gpuVector findRayDirectionDevice( curandState* randNumStates );


__device__ void rayDirection_cellFaceDevice( curandState* randNumStates,
                                             const gpuIntVector& origin,
                                             const gpuIntVector& indexOrder, 
                                             const gpuIntVector& signOrder,
                                             const int iRay,
                                             gpuVector& directionVector,
                                             double& cosTheta);
                            
__device__ gpuVector rayLocationDevice( curandState* randNumStates,
                                      const gpuIntVector origin,
                                      const double DyDx, 
                                      const double DzDx,
                                      const bool useCCRays);
                                      
__device__ void rayLocation_cellFaceDevice( curandState* randNumStates,
                                            const gpuIntVector& origin,
                                            const gpuIntVector &indexOrder, 
                                            const gpuIntVector &shift, 
                                            const double &DyDx, 
                                            const double &DzDx,
                                            gpuVector& location);


__device__ bool has_a_boundaryDevice(const gpuIntVector &c, 
                                     const GPUGridVariable<int>& celltype, 
                                     BoundaryFaces &boundaryFaces);


__device__ void findStepSizeDevice(int step[],
                                   bool sign[],
                                   const gpuVector& inv_direction_vector);
                                 
__device__ void reflect(double& fs,
                        gpuIntVector& cur,
                        gpuIntVector& prevCell,
                        const double abskg,
                        bool& in_domain,
                        int& step,
                        bool& sign,
                        double& ray_direction);
template<class T>                                                          
__device__ void updateSumIDevice ( gpuVector& ray_direction,
                                   gpuVector& ray_location,
                                   const gpuIntVector& origin,
                                   const gpuVector& Dx,
                                   const GPUGridVariable< T >&  sigmaT4OverPi,
                                   const GPUGridVariable< T >& abskg,
                                   const GPUGridVariable<int>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags);

__device__ double randDblExcDevice(curandState* randNumStates);


__device__ double randDevice(curandState* randNumStates);


__device__ unsigned int hashDevice(unsigned int a);

//______________________________________________________________________
//
//__________________________________
//  returns gpuVector * scalar
inline HOST_DEVICE gpuVector operator*(const gpuVector & a, double b) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}
//__________________________________
//  returns gpuVector * scalar
inline HOST_DEVICE gpuVector operator*(double b, const gpuVector & a) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}

//__________________________________
//  returns gpuVector * gpuVector
inline HOST_DEVICE gpuVector operator*(const gpuVector& a, const gpuVector& b) {
  return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//__________________________________
//  returns gpuVector/scalar
inline HOST_DEVICE gpuVector operator/(const gpuVector & a, double b) {
  b = 1.0f / b;
  return a*b;
}
//__________________________________
//  returns scalar/gpuVector
inline HOST_DEVICE gpuVector operator/(double a, const gpuVector& b){
  return make_double3(a/b.x, a/b.y, a/b.z);
}

//__________________________________
//  returns abs
inline HOST_DEVICE gpuVector Abs(const gpuVector& v){

  double x = v.x < 0 ? -v.x:v.x;
  double y = v.y < 0 ? -v.y:v.y;
  double z = v.z < 0 ? -v.z:v.z;
  return make_double3(x,y,z);
}

} //end namespace Uintah

#endif
