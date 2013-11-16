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

#include <Core/Geometry/GPUVector.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <sci_defs/cuda_defs.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>

namespace Uintah {

using namespace std;
using namespace SCIRun;


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
  gpuVector dx;             // cell spacing
  gpuIntVector lo;          // cell low index not including extra or ghost cells
  gpuIntVector hi;          // cell high index not including extra or ghost cells
  gpuIntVector nCells;      // number of cells in each dir
  int ID;                   // patch ID
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
  int   nRaySteps;            // number of ray steps taken
  
};


enum DIR {X=0, Y=1, Z=2, NONE=-9};
//           -x      +x       -y       +y     -z     +z
enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};

void launchRayTraceKernel(dim3 dimGrid,
                          dim3 dimBlock,
                          int matlIndex,
                          patchParams patch,
                          const gpuIntVector domainLo,
                          const gpuIntVector domainHi,
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
                               const gpuIntVector domainLo,
                               const gpuIntVector domainHi,
                               curandState randNumStates,
                               RMCRT_flags RT_flags,
                               varLabelNames* labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw);
                               
__device__ gpuVector findRayDirectionDevice(curandState* randNumStates,
                                          const bool isSeedRandom,
                                          const gpuIntVector origin,
                                          const int iRay,
                                          const int tidX);
                                    
__device__ gpuVector rayLocationDevice( curandState* randNumStates,
                                      const gpuIntVector origin,
                                      const double DyDx, 
                                      const double DzDx,
                                      const bool useCCRays);

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
                                                          
__device__ void updateSumIDevice ( gpuVector& ray_direction,
                                   gpuVector& ray_location,
                                   const gpuIntVector& origin,
                                   const gpuVector& Dx,
                                   const GPUGridVariable<double>&  sigmaT4OverPi,
                                   const GPUGridVariable<double>& abskg,
                                   const GPUGridVariable<int>& celltype,
                                   double& sumI,
                                   curandState* randNumStates,
                                   RMCRT_flags RT_flags);

__device__ bool containsCellDevice(const gpuIntVector& domainLow,
                                   const gpuIntVector& domainHigh,
                                   const gpuIntVector& cell,
                                   const int& face);


__device__ double randDblExcDevice(curandState* randNumStates);


__device__ double randDevice(curandState* randNumStates);


__device__ unsigned int hashDevice(unsigned int a);

#if 1
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
#endif

} //end namespace Uintah

#endif
