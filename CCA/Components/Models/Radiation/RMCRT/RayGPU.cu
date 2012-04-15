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
                      Task::WhichDW which_sigmaT4_dw)
{
  // set the CUDA context
  cudaError_t retVal;
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device));

  const Level* level = getLevel(patches);

  // Determine the size of the domain.
  IntVector domainLo, domainHi;
  IntVector domainLo_EC, domainHi_EC;

  level->findInteriorCellIndexRange(domainLo, domainHi);     // excluding extraCells
  level->findCellIndexRange(domainLo_EC, domainHi_EC);       // including extraCells

  DataWarehouse* abskg_dw = new_dw->getOtherDataWarehouse(which_abskg_dw);
  DataWarehouse* sigmaT4_dw = new_dw->getOtherDataWarehouse(which_sigmaT4_dw);

  constCCVariable<double> sigmaT4Pi;
  constCCVariable<double> abskg;
  abskg_dw->getRegion(abskg, d_abskgLabel, d_matl, level, domainLo_EC, domainHi_EC);
  sigmaT4_dw->getRegion(sigmaT4Pi, d_sigmaT4_label, d_matl, level, domainLo_EC, domainHi_EC);

  // Single material now, but can't assume 0, need the specific ARCHES or ICE material here
  int matl = matls->getVector().front();
  int numPatches = patches->size();

  // requires and computes on device
  double* d_absk = NULL;
  double* d_sigmaT4 = NULL;
  double* d_divQ = NULL;

  // patch loop
  for (int p = 0; p < numPatches; p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTraceGPU");

    d_absk    = _gpuScheduler->getDeviceRequiresPtr(d_abskgLabel, matl, patch);
    d_sigmaT4 = _gpuScheduler->getDeviceRequiresPtr(d_sigmaT4_label, matl, patch);
    d_divQ    = _gpuScheduler->getDeviceComputesPtr(d_divQLabel, matl, patch);

    // Calculate the memory block size
    IntVector nec = patch->getExtraCells();
    IntVector l = patch->getCellLowIndex();
    IntVector h = patch->getCellHighIndex();
    h += nec;

    IntVector divQSize = _gpuScheduler->getDeviceComputesSize(d_divQLabel, matl, patch);
    int xdim = divQSize.x();
    int ydim = divQSize.y();
    int zdim = divQSize.z();

    Vector dcell = patch->dCell(); // cell spacing
    const double3 cellSpacing = make_double3(dcell.x(), dcell.y(), dcell.z());

    IntVector pLow;
    IntVector pHigh;
    level->findInteriorCellIndexRange(pLow, pHigh);
    int cellIndexRange = pHigh[0] - pLow[0];

    // Domain extents used by the kernel to prevent out of bounds accesses.
    const uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
    const uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());
    const uint3 domainSize = make_uint3(xdim, ydim, zdim);

    int xBlocks = xdim / 8;
    if (xdim % 8 != 0) {
      xBlocks++;
    }
    int yBlocks = ydim / 8;
    if (ydim % 8 != 0) {
      yBlocks++;
    }
    dim3 totalBlocks(xBlocks, yBlocks);

    int tpbX = 8;
    int tpbY = 8;
    int tpbZ = 1;
    dim3 threadsPerBlock(tpbX, tpbY, tpbZ);

    // setup random number generator states on the device
    curandState* globalDevStates;
    int numStates = totalBlocks.x * totalBlocks.y * tpbX * tpbY * tpbZ;
    CUDA_SAFE_CALL( cudaMalloc((void**)&globalDevStates, numStates * sizeof(curandState)) );

    // setup and launch kernel
    cudaStream_t* stream = _gpuScheduler->getCudaStream(device);
    cudaEvent_t* event = _gpuScheduler->getCudaEvent(device);
    rayTraceKernel<<< totalBlocks, threadsPerBlock, 0, *stream >>>(domainLow,
                                                                   domainHigh,
                                                                   domainSize,
                                                                   cellSpacing,
                                                                   cellIndexRange,
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

    // Kernel error checking (for now)
    retVal = cudaGetLastError();
    if (retVal != cudaSuccess) {
      fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
      exit(-1);
    }

    _gpuScheduler->requestD2HCopy(d_divQLabel, matl, patch, stream, event);

    CUDA_SAFE_CALL( cudaFree(globalDevStates) );

  }  //end patch loop
}  // end GPU ray trace method


//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel(const uint3 domainLow,
                               const uint3 domainHigh,
                               const uint3 domainSize,
                               const double3 cellSpacing,
                               int cellIndexRange,
                               double* __restrict__ device_abskg,
                               double* __restrict__ device_sigmaT4,
                               double* __restrict__ device_divQ,
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

  // Get the size of the data block in which the variables reside.
  // This is essentially the stride in the index calculations.
  int dx = domainSize.x;
  int dy = domainSize.y;

  // initialize device RNG states
  curand_init(hashDevice(tidX), tidX, 0, &globalDevStates[tidX]);

  // GPU equivalent to GridIterator loop
  if (tidX > 0 && tidY > 0 && tidX < domainHigh.x && tidY < domainHigh.y) { // domain boundary check
    for (int z = domainLow.z; z < domainHigh.z; z++) { // loop through z slices

      // calculate the index for individual threads
      int idx = INDEX3D(dx,dy,tidX,tidY,z);

      uint3 origin = make_uint3(tidX, tidY, z);  // for each thread
      double sumI = 0;

      //_______________________________________________________________________
      // ray loop
      for (int iRay = 0; iRay < numRays; iRay++) {

//        // TODO Seed on device
//        if (isSeedRandom == false) {
//          _mTwister.seed((i + j + k) * iRay + 1);
//        }

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

        updateSumIDevice(domainLow, domainHigh, domainSize, origin, cellSpacing, inv_direction_vector,
                         ray_location, device_sigmaT4, device_abskg, threshold, &sumI);

      } // end ray loop

      //__________________________________
      //  Compute divQ
      device_divQ[idx] = 4.0 * M_PI * device_abskg[idx] * (device_sigmaT4[idx] - (sumI / numRays)) * 0.95;

    } // end z-slice loop
  }  // end domain boundary check
}  // end ray trace kernel


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ void updateSumIDevice(const uint3& domainLow,
                                 const uint3& domainHigh,
                                 const uint3& domainSize,
                                 const uint3& origin,
                                 const double3& cellSpacing,
                                 const double3& inv_direction_vector,
                                 const double3& ray_location,
                                 double* __restrict__ device_sigmaT4,
                                 double* __restrict__ device_abskg,
                                 double threshold,
                                 double* sumI)
{
  // calculate the thread indices
  int tidX = threadIdx.x + blockIdx.x * blockDim.x;
  int tidY = threadIdx.y + blockIdx.y * blockDim.y;

  // Get the size of the data block in which the variables reside.
  // This is essentially the stride in the index calculations.
  int dx = domainSize.x;
  int dy = domainSize.y;;

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
  while (intensity > threshold) { // threshold while loop
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

      in_domain = containsCellDevice(domainLow, domainHigh, cur, face);

      //__________________________________
      //  Update the ray location
      double optical_thickness_prev = optical_thickness;
      int prev_index = INDEX3D(dx,dy,prevCell.x,prevCell.y,prevCell.z) + (dx*dy);
      optical_thickness += cellSpacing.x * device_abskg[prev_index] * disMin;
      *sumI += device_sigmaT4[prev_index] * ( exp(-optical_thickness_prev) - exp(-optical_thickness) ) * fs;

    } // end domain while loop

    intensity = exp(-optical_thickness);
    int cur_index = INDEX3D(dx,dy,cur.x,cur.y,cur.z) + (dx*dy);
    *sumI += device_abskg[cur_index] * device_sigmaT4[cur_index] * intensity;
    intensity = intensity * (1 - device_abskg[cur_index]);

    //__________________________________
    //  Reflections
    if (intensity > threshold) {

      ++nReflect;
      fs = fs * (1 - device_abskg[cur_index]);

      //put cur back inside the domain
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
__device__ bool containsCellDevice(const uint3& domainLow,
                                   const uint3& domainHigh,
                                   const uint3& cell,
                                   const int& face)
{
  switch (face) {
    case 0 :
      return domainLow.x <= cell.x && domainHigh.x > cell.x;
    case 1 :
      return domainLow.y <= cell.y && domainHigh.y > cell.y;
    case 2 :
      return domainLow.z <= cell.z && domainHigh.z > cell.z;
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
