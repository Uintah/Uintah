/*

 The MIT License

 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
 University of Utah.

 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 */

#ifndef RAY_GPU_CUH
#define RAY_GPU_CUH

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NOTE ON DEVICE CODE:
 *
 *  CUDA does not have a linker for device code, therefore all called __device__ functions must
 *  be visible from the calling function. This means the called function needs to be either in
 *  the same file, or in a file included by the file from which the function is called.
 */

__global__ void rayTraceKernel(const uint3 domainLow,
                               const uint3 domainHigh,
                               const uint3 domainSize,
                               const double3 cellSpacing,
                               int cellIndexRange,
                               double* device_abskg,
                               double* device_sigmaT4,
                               double* device_divQ,
                               bool virtRad,
                               bool isSeedRandom,
                               bool ccRays,
                               int numRays,
                               double viewAngle,
                               double threshold,
                               unsigned long int* size,
                               curandState* globalDevStates);


__device__ void updateSumIDevice(const uint3& domainLow,
                                 const uint3& domainHigh,
                                 const uint3& origin,
                                 const double3& cellSpacing,
                                 const double3& inv_direction_vector,
                                 const double3& ray_location,
                                 double* device_sigmaT4,
                                 double* device_abskg,
                                 double threshold,
                                 unsigned long int* size,
                                 double* sumI);


__device__ bool containsCellDevice(const uint3& domainLow,
                                   const uint3& domainHigh,
                                   const uint3& cell,
                                   const int& face);


__device__ double randDblExcDevice(curandState* globalState);


__device__ double randDevice(curandState* globalState);


__device__ unsigned int hashDevice(unsigned int a);

#ifdef __cplusplus
}
#endif

#endif
