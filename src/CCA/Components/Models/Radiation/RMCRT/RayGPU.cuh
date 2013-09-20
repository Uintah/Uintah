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

#ifndef RAY_GPU_CUH
#define RAY_GPU_CUH

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>

namespace Uintah {
using namespace std;


HOST_DEVICE struct varLabelNames{
  const char* divQ;
  const char* abskg;
  const char* sigmaT4;
  const char* celltype;
  const char* VRFlux;
  const char* boundFlux;
  const char* radVolQ;
};

HOST_DEVICE struct patchParams{
  double3 dx;             // cell spacing
  uint3 lo;               // cell low index not including extra or ghost cells
  uint3 hi;               // cell high index not including extra or ghost cells
  uint3 nCells;           // number of cells in each dir
  int ID;                 // patch ID
};

HOST_DEVICE struct RMCRT_flags{
  bool modifies_divQ;
  bool virtRad;
  bool solveDivQ;
  bool allowReflect;
  bool solveBoundaryFlux;
  bool isSeedRandom;
  bool CCRays;
  
  double sigma;               // StefanBoltzmann constant
  double sigmaScat;           // scattering coefficient
  double threshold;
  
  int   nDivQRays;            // number of rays per cell used to compute divQ
  int   nRadRays;             // number of rays for virtual radiometer
  int   nFluxRays;            // number of boundary flux rays
  
};

void launchRayTraceKernel(dim3 dimGrid,
                          dim3 dimBlock,
                          int matlIndex,
                          patchParams patch,
                          const uint3 domainLo,
                          const uint3 domainHi,
                          curandState* randNumStates,
                          cudaStream_t* stream,
                          RMCRT_flags RT_flags,                               
                          varLabelNames labelNames,
                          GPUDataWarehouse* abskg_gdw,
                          GPUDataWarehouse* sigmaT4_gdw,
                          GPUDataWarehouse* celltype_gdw,
                          GPUDataWarehouse* old_gdw,
                          GPUDataWarehouse* new_gdw);


__global__ void rayTraceKernel(dim3 dimGrid,
                               dim3 dimBlock,
                               int matlIndex,
                               patchParams patch,
                               const uint3 domainLo,
                               const uint3 domainHi,
                               curandState* randNumStates,
                               RMCRT_flags RT_flags,
                               varLabelNames labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw);
                               
__device__ double3 findRayDirectionDevice(curandState* randNumStates,
                                          const bool isSeedRandom,
                                          const uint3 origin,
                                          const int iRay,
                                          const int tidX);
                                    
__device__ double3 rayLocationDevice( curandState* randNumStates,
                                      const uint3 origin,
                                      const double DyDx, 
                                      const double DzDx,
                                      const bool useCCRays);

__device__ void findStepSizeDevice(int step[],
                                   bool sign[],
                                   const double3& inv_direction_vector);
                                   
__device__ void updateSumIDevice(const uint3& domainLow,
                                 const uint3& domainHigh,
                                 const uint3& domainSize,
                                 const uint3& origin,
                                 const double3& cellSpacing,
                                 const double3& inv_direction_vector,
                                 const double3& ray_location,
                                 double* device_sigmaT4,
                                 double* device_abskg,
                                 double* threshold,
                                 double* sumI);


__device__ bool containsCellDevice(const uint3& domainLow,
                                   const uint3& domainHigh,
                                   const uint3& cell,
                                   const int& face);


__device__ double randDblExcDevice(curandState* randNumStates);


__device__ double randDevice(curandState* randNumStates);


__device__ unsigned int hashDevice(unsigned int a);




//______________________________________________________________________
//
// returns a - b
inline HOST_DEVICE double3 operator-(const double3 & a, const double3 & b) {
  return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}
//__________________________________
//  returns a + b
inline HOST_DEVICE double3 operator+(const double3 & a, const double3 & b) {
  return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}
//__________________________________
//  return -a
inline HOST_DEVICE double3 operator-(const double3 & a) {
  return make_double3(-a.x,-a.y,-a.z);
}
//__________________________________
//  returns double3 * scalar
inline HOST_DEVICE double3 operator*(const double3 & a, double b) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}
//__________________________________
//  returns double3 * scalar
inline HOST_DEVICE double3 operator*(double b, const double3 & a) {
  return make_double3(a.x*b, a.y*b, a.z*b);
}
//__________________________________
//  returns double3/scalar
inline HOST_DEVICE double3 operator/(const double3 & a, double b) {
  b = 1.0f / b;
  return a*b;
}
//__________________________________
//  returns scalar/double3
inline HOST_DEVICE double3 operator/(double a, const double3& b){
  return make_double3(a/b.x, a/b.y, a/b.z);
}

//______________________________________________________________________
//
// returns a - b
inline HOST_DEVICE uint3 operator-(const uint3 & a, const uint3 & b) {
  return make_uint3(a.x-b.x, a.y-b.y, a.z-b.z);
}
//__________________________________
//  returns a + b
inline HOST_DEVICE uint3 operator+(const uint3 & a, const uint3 & b) {
  return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z);
}
//__________________________________
//  return -a
inline HOST_DEVICE uint3 operator-(const uint3 & a) {
  return make_uint3(-a.x,-a.y,-a.z);
}
//__________________________________
//  returns uint3 * scalar
inline HOST_DEVICE uint3 operator*(const uint3 & a, int b) {
  return make_uint3(a.x*b, a.y*b, a.z*b);
}
//__________________________________
//  returns uint3 * scalar
inline HOST_DEVICE uint3 operator*(int b, const uint3 & a) {
  return make_uint3(a.x*b, a.y*b, a.z*b);
}
//__________________________________
//  returns uint3/scalar
inline HOST_DEVICE uint3 operator/(const uint3 & a, int b) {
  b = 1.0f / b;
  return a*b;
}
//__________________________________
//  returns scalar/uint3
inline HOST_DEVICE uint3 operator/(int a, const uint3& b){
  return make_uint3(a/b.x, a/b.y, a/b.z);
}


} //end namespace Uintah

#endif
