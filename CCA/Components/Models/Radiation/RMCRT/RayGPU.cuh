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
                          curandState* globalDevRandStates,
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
                               curandState* globalDevRandStates,
                               RMCRT_flags RT_flags,
                               varLabelNames labelNames,
                               GPUDataWarehouse* abskg_gdw,
                               GPUDataWarehouse* sigmaT4_gdw,
                               GPUDataWarehouse* celltype_gdw,
                               GPUDataWarehouse* old_gdw,
                               GPUDataWarehouse* new_gdw);


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


__device__ double randDblExcDevice(curandState* globalState);


__device__ double randDevice(curandState* globalState);


__device__ unsigned int hashDevice(unsigned int a);


} //end namespace Uintah

#endif
