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

__global__ void rayTraceKernel(uint3 domainLow,
                               uint3 domainHigh,
                               uint3 domainSize,
                               double* d_absk,
                               double* d_sigmaT4,
                               double* d_divQ,
                               double* d_VRFlux);

__device__ bool containsCellKernel(const int3& low,
                                   const int3& high,
                                   const int3& cell,
                                   const int&  face);

__device__ void updateSumIKernel(const double3& inv_direction_vector,
                                 const double3& ray_location,
                                 const int3& origin,
                                 const double3& Dx,
                                 const int3& domainLo,
                                 const int3& domainHi,
                                 double* sigmaT4Pi,
                                 double* abskg,
                                 unsigned long int& size,
                                 double threshold,
                                 double& sumI);

#ifdef __cplusplus
}
#endif

#endif
