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

//----- RayGPU.cu ----------------------------------------------
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <Core/Grid/DbgOutput.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAY_GPU", false);


//---------------------------------------------------------------------------
// Method: The GPU ray tracer - setup and invoke ray trace kernel
//---------------------------------------------------------------------------
void Ray::rayTraceGPU(const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      int device,
                      bool modifies_divQ,
                      Task::WhichDW which_abskg_dw,
                      Task::WhichDW which_sigmaT4_dw,
                      Task::WhichDW which_celltype_dw)
{
  // set the CUDA device and context
  CUDA_RT_SAFE_CALL( cudaSetDevice(device) );

  // Single material now, but can't assume 0, need the specific ARCHES or ICE material here
  int matl = matls->getVector().front();
  int numPatches = patches->size();

  const Level* level = getLevel(patches);
  IntVector dLo, dHi;
  level->findInteriorCellIndexRange(dLo, dHi);
  uint3 domainLo = make_uint3(dLo.x(), dLo.y(), dLo.z());
  uint3 domainHi = make_uint3(dLo.x(), dHi.y(), dHi.z());

  // requires and computes on device
  double* d_absk = NULL;
  double* d_sigmaT4 = NULL;
  double* d_divQ = NULL;

  // patch loop
  for (int p = 0; p < numPatches; p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    // pointers to device-side grid-variables
    d_absk    = _gpuScheduler->getDeviceRequiresPtr(d_abskgLabel, matl, patch);
    d_sigmaT4 = _gpuScheduler->getDeviceRequiresPtr(d_sigmaT4_label, matl, patch);
    d_divQ    = _gpuScheduler->getDeviceComputesPtr(d_divQLabel, matl, patch);

    // Calculate the memory block size
    IntVector nec = patch->getExtraCells();
    IntVector l = patch->getCellLowIndex();
    IntVector h = patch->getCellHighIndex();

    IntVector divQSize = _gpuScheduler->getDeviceComputesSize(d_divQLabel, matl, patch);
    int xdim = divQSize.x();
    int ydim = divQSize.y();
    int zdim = divQSize.z();

    // get the cell spacing and convert to CUDA vector type
    Vector dcell = patch->dCell();
    const double3 cellSpacing = make_double3(dcell.x(), dcell.y(), dcell.z());

    // Patch extents used by the kernel to prevent out of bounds accesses.
    const uint3 patchLo = make_uint3(l.x(), l.y(), l.z());
    const uint3 patchHi = make_uint3(h.x(), h.y(), h.z());
    const uint3 patchSize = make_uint3(xdim, ydim, zdim);

    int xBlocks = xdim / 8;
    if (xdim % 8 != 0) {
      xBlocks++;
    }
    int yBlocks = ydim / 8;
    if (ydim % 8 != 0) {
      yBlocks++;
    }
    dim3 dimGrid(xBlocks, yBlocks); // grid dimensions

    int tpbX = 8;
    int tpbY = 8;
    int tpbZ = 1;
    dim3 dimBlock(tpbX, tpbY, tpbZ); // block dimensions

    // setup random number generator states on the device, 1 for each thread
    curandState* globalDevStates;
    int numStates = dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y * dimBlock.z;
    CUDA_RT_SAFE_CALL( cudaMalloc((void**)&globalDevStates, numStates * sizeof(curandState)) );

    // setup and launch kernel
    cudaStream_t* stream = _gpuScheduler->getCudaStream(device);
    rayTraceKernel<<< dimGrid, dimBlock, 0, *stream >>>(patchLo,
                                                        patchHi,
                                                        patchSize,
                                                        domainLo,
                                                        domainHi,
                                                        cellSpacing,
                                                        d_absk,
                                                        d_sigmaT4,
                                                        d_divQ,
                                                        this->_virtRad,
                                                        this->_isSeedRandom,
                                                        this->_CCRays,
                                                        this->_NoOfRays,
                                                        this->_viewAng,
                                                        this->_Threshold,
                                                        globalDevStates);


    // get updated divQ back into host memory
    cudaEvent_t* event = _gpuScheduler->getCudaEvent(device);
    _gpuScheduler->requestD2HCopy(d_divQLabel, matl, patch, stream, event);

    // free device-side RNG states
    CUDA_RT_SAFE_CALL( cudaFree(globalDevStates) );

  }  //end patch loop
}  // end GPU ray trace method


//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel(const uint3 patchLo,
                               const uint3 patchHi,
                               const uint3 patchSize,
                               const uint3 domainLo,
                               const uint3 domainHi,
                               const double3 cellSpacing,
                               double* device_abskg,
                               double* device_sigmaT4,
                               double* device_divQ,
                               bool virtRad,
                               bool isSeedRandom,
                               bool ccRays,
                               int numRays,
                               double viewAngle,
                               double threshold,
                               curandState* globalDevStates)
{
  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y;

  // Get the extents of the data block in which the variables reside.
  // This is essentially the stride in the index calculations.
  int dx = patchSize.x;
  int dy = patchSize.y;

  // GPU equivalent of GridIterator loop - calculate sets of rays per thread
  if (tidX >= patchLo.x && tidY >= patchLo.y && tidX <= patchHi.x && tidY <= patchHi.y) { // patch boundary check
    for (int z = patchLo.z; z <= patchHi.z; z++) { // loop through z slices

      // calculate the index for individual threads
      int idx = INDEX3D(dx,dy,tidX,tidY,z);

      uint3 origin = make_uint3(tidX, tidY, z);  // for each thread
      double sumI = 0;

      //_______________________________________________________________________
      // ray loop
      #pragma unroll
      for (int iRay = 0; iRay < numRays; iRay++) {

        // initialize device RNG states
        if (isSeedRandom == false) {
          curand_init(hashDevice(tidX), tidX, 0, &globalDevStates[tidX]);
        }

        // for explanation see: http://www.cgafaq.info/wiki/Random_Points_On_Sphere
        double plusMinus_one = 2 * randDblExcDevice(globalDevStates) - 1;
        double r = sqrt(1 - plusMinus_one * plusMinus_one);    // Radius of circle at z
        double theta = 2 * M_PI * randDblExcDevice(globalDevStates);      // Uniform between 0-2Pi

        // Convert to cartesian coordinates
        double3 direction_vector;
        direction_vector.x = r * cos(theta);
        direction_vector.y = r * sin(theta);
        direction_vector.z = plusMinus_one;

        double3 inv_direction_vector;
        inv_direction_vector.x = 1.0 / direction_vector.x;
        inv_direction_vector.y = 1.0 / direction_vector.y;
        inv_direction_vector.z = 1.0 / direction_vector.z;

        double DyDxRatio = cellSpacing.y / cellSpacing.x;  //noncubic
        double DzDxRatio = cellSpacing.z / cellSpacing.x;  //noncubic

        double3 ray_location;

        if (ccRays) {
          ray_location.x = origin.x + 0.5;
          ray_location.y = origin.y + 0.5 * DyDxRatio;  //noncubic
          ray_location.z = origin.z + 0.5 * DzDxRatio;  //noncubic
        } else {
          ray_location.x = origin.x + randDevice(globalDevStates);
          ray_location.y = origin.y + randDevice(globalDevStates) * DyDxRatio;  //noncubic
          ray_location.z = origin.z + randDevice(globalDevStates) * DzDxRatio;  //noncubic
        }

        updateSumIDevice(domainLo, domainHi, patchSize, origin, cellSpacing, inv_direction_vector,
                         ray_location, device_sigmaT4, device_abskg, &threshold, &sumI);

      }  // end ray loop

      //__________________________________
      //  Compute divQ
      device_divQ[idx] = 4.0 * M_PI * device_abskg[idx] * (device_sigmaT4[idx] - (sumI / numRays));

    } // end z-slice loop
  }  // end domain boundary check
}  // end ray trace kernel


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ void updateSumIDevice(const uint3& domainLo,
                                 const uint3& domainHi,
                                 const uint3& patchSize,
                                 const uint3& origin,
                                 const double3& cellSpacing,
                                 const double3& inv_direction_vector,
                                 const double3& ray_location,
                                 double* device_sigmaT4,
                                 double* device_abskg,
                                 double* threshold,
                                 double* sumI)
{
  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y;

  // Get the size of the data block in which the variables reside.
  // This is essentially the stride in the index calculations.
  int dx = patchSize.x;
  int dy = patchSize.y;;

  uint3 cur = origin;
  uint3 prevCell = cur;

  // Step and sign for ray marching, gives +1 or -1 based on sign
  int step[3];
  bool sign[3];

  // unrolled
  if (inv_direction_vector.x > 0) {
    step[0] = 1;
    sign[0] = 1;
  } else {
    step[0] = -1;
    sign[0] = 0;
  }

  if (inv_direction_vector.y > 0) {
    step[1] = 1;
    sign[1] = 1;
  } else {
    step[1] = -1;
    sign[1] = 0;
  }

  if (inv_direction_vector.z > 0) {
    step[2] = 1;
    sign[2] = 1;
  } else {
    step[2] = -1;
    sign[2] = 0;
  }

  double DyDxRatio = cellSpacing.y / cellSpacing.x;  //noncubic
  double DzDxRatio = cellSpacing.z / cellSpacing.x;  //noncubic

  double tMaxX = (origin.x + sign[0] - ray_location.x) * inv_direction_vector.x;
  double tMaxY = (origin.y + sign[1] * DyDxRatio - ray_location.y) * inv_direction_vector.y;
  double tMaxZ = (origin.z + sign[2] * DzDxRatio - ray_location.z) * inv_direction_vector.z;

  //Length of t to traverse one cell
  double tDeltaX = abs(inv_direction_vector.x);
  double tDeltaY = abs(inv_direction_vector.y) * DyDxRatio;
  double tDeltaZ = abs(inv_direction_vector.z) * DzDxRatio;
  double tMax_prev = 0;
  bool in_domain = true;

  //Initializes the following values for each ray
  double intensity = 1.0;
  double fs = 1.0;
  double optical_thickness = 0;

  // begin ray tracing
  int nReflect = 0;  // Number of reflections that a ray has undergone
  while (intensity > *threshold) { // threshold while loop
    int face = -9;
    while (in_domain) {
      prevCell = cur;
      double disMin = -9;  // Common variable name in ray tracing. Represents ray segment length.

      //__________________________________
      //  Determine which cell the ray will enter next
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          cur.x = cur.x + step[0];
          disMin = tMaxX - tMax_prev;
          tMax_prev = tMaxX;
          tMaxX = tMaxX + tDeltaX;
          face = 0;
        } else {
          cur.z = cur.z + step[2];
          disMin = tMaxZ - tMax_prev;
          tMax_prev = tMaxZ;
          tMaxZ = tMaxZ + tDeltaZ;
          face = 2;
        }
      } else {
        if (tMaxY < tMaxZ) {
          cur.y = cur.y + step[1];
          disMin = tMaxY - tMax_prev;
          tMax_prev = tMaxY;
          tMaxY = tMaxY + tDeltaY;
          face = 1;
        } else {
          cur.z = cur.z + step[2];
          disMin = tMaxZ - tMax_prev;
          tMax_prev = tMaxZ;
          tMaxZ = tMaxZ + tDeltaZ;
          face = 2;
        }
      }

      in_domain = containsCellDevice(domainLo, domainHi, cur, face);

      //__________________________________
      //  Update the ray location
      double optical_thickness_prev = optical_thickness;
      int prev_index = INDEX3D(dx,dy,prevCell.x,prevCell.y,prevCell.z) + (dx*dy);
      optical_thickness += cellSpacing.x * device_abskg[prev_index] * disMin;
      // device_sigmaT4[idx] always 0.3183314161909468?
      *sumI += device_sigmaT4[prev_index] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;

    } // end domain while loop

    intensity = exp(-optical_thickness);
    int cur_index = INDEX3D(dx,dy,cur.x,cur.y,cur.z) + (dx*dy);
    *sumI += device_abskg[cur_index] * device_sigmaT4[cur_index] * intensity;
    intensity = intensity * (1 - device_abskg[cur_index]);

    //__________________________________
    //  Reflections
    if (intensity > *threshold) {

      ++nReflect;
      fs = fs * (1 - device_abskg[cur_index]);

      // put cur back inside the domain
      cur = prevCell;

      // apply reflection condition
      step[face] *= -1;                        // begin stepping in opposite direction
      sign[face] = (sign[face] == 1) ? 0 : 1;  // swap sign from 1 to 0 or vice versa

      in_domain = true;

    }  // end if reflection
  }  // end threshold while loop.
}  // end of updateSumI function


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ bool containsCellDevice(const uint3& domainLo,
                                   const uint3& domainHi,
                                   const uint3& cell,
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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    double val = curand(&localState);
    globalState[tid] = localState;

    return (double)val * (1.0/4294967295.0);
}


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ double randDblExcDevice(curandState* globalState)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[tid];
    double val = curand_uniform(&localState);
    globalState[tid] = localState;

    return ( (double)val + 0.5 ) * (1.0/4294967296.0);
}


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__host__ __device__ unsigned int hashDevice(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);

    return a;
}
