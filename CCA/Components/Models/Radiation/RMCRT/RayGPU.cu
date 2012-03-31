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
#include <CCA/Components/Models/Radiation/RMCRT/MersenneTwister.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/Models/Radiation/RMCRT/RayGPU.cuh>
#include <Core/Grid/DbgOutput.h>

#include <sci_defs/cuda_defs.h>

using namespace Uintah;
using namespace std;

static DebugStream dbggpu("RAY_GPU", false);


//---------------------------------------------------------------------------
// Method: The GPU ray tracer
//---------------------------------------------------------------------------
void Ray::rayTraceGPU(const ProcessorGroup* pc,
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
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );

  initMTRandGPU();

  const Level* level = getLevel(patches);
  MTRand _mTwister;

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

  double start = clock();
  //--------------------------------------------------

  // Single material now, but can't assume 0, need the specific ARCHES or ICE material here
  int matl = matls->getVector().front();
  int numPatches = patches->size();

  // requires and computes
  double* d_absk = NULL;
  double* d_sigmaT4 = NULL;
  double* d_divQ = NULL;
  double* d_VRFlux = NULL;

  // patch loop
  for (int p = 0; p < numPatches; p++) {

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbggpu, "Doing Ray::rayTrace");

    d_absk    = _gpuScheduler->getDeviceRequiresPtr(d_abskgLabel, matl, patch);
    d_sigmaT4 = _gpuScheduler->getDeviceRequiresPtr(d_sigmaT4_label, matl, patch);
    d_divQ    = _gpuScheduler->getDeviceComputesPtr(d_divQLabel, matl, patch);
    d_VRFlux  = _gpuScheduler->getDeviceComputesPtr(d_VRFluxLabel, matl, patch);

    // Calculate the memory block size
    IntVector nec = patch->getExtraCells();
    IntVector l = patch->getCellLowIndex();
    IntVector h = patch->getCellHighIndex();
    h += nec;
    IntVector divQSize = _gpuScheduler->getDeviceRequiresSize(d_divQLabel, matl, patch);
    IntVector VRFluxSize = _gpuScheduler->getDeviceRequiresSize(d_VRFluxLabel, matl, patch);
    int xdim = divQSize.x(), ydim = divQSize.y();

    unsigned long int size = 0;                        // current size of PathIndex
    Vector dcell = patch->dCell();                     // cell spacing
    double3 Dx = make_double3(dcell.x(), dcell.y(), dcell.z());

    // Domain extents used by the kernel to prevent out of bounds accesses.
    uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
    uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());
    uint3 domainSize = make_uint3(divQSize.x(), divQSize.y(), divQSize.z());

    int xBlocks = xdim / 8;
    if (xdim % 8 != 0) { xBlocks++; }
    int yBlocks = ydim / 8;
    if (ydim % 8 != 0) { yBlocks++; }
    dim3 totalBlocks(xBlocks, yBlocks);
    dim3 threadsPerBlock(8, 8, 1);

    // setup and launch kernel
    cudaStream_t* stream = _gpuScheduler->getCudaStream(device);
    cudaEvent_t* event   = _gpuScheduler->getCudaEvent(device);
    rayTraceKernel<<< totalBlocks, threadsPerBlock, 0, *stream >>>(domainLow,
                                                                   domainHigh,
                                                                   domainSize,
                                                                   d_absk,
                                                                   d_sigmaT4,
                                                                   d_divQ,
                                                                   d_VRFlux);

    // Kernel error checking (for now)
    retVal = cudaGetLastError();
    if (retVal != cudaSuccess) {
      fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
      exit(-1);
    }

    _gpuScheduler->requestD2HCopy(d_divQLabel, matl, patch, stream, event);
    _gpuScheduler->requestD2HCopy(d_VRFluxLabel, matl, patch, stream, event);

    double end = clock();
    double efficiency = size / ((end - start) / CLOCKS_PER_SEC);
    if (patch->getGridIndex() == 0) {
      cout << endl;
      cout << " RMCRT REPORT: Patch 0" << endl;
      cout << " Used " << (end - start) * 1000 / CLOCKS_PER_SEC << " milliseconds of CPU time. \n"
      << endl;  // Convert time to ms
      cout << " Size: " << size << endl;
      cout << " Efficiency: " << efficiency << " steps per sec" << endl;
      cout << endl;
    }
  }  //end patch loop
}  // end GPU ray trace method


//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
void Ray::initMTRandGPU()
{

}


//---------------------------------------------------------------------------
// Kernel: The GPU ray tracer kernel
//---------------------------------------------------------------------------
__global__ void rayTraceKernel(uint3 domainLow,
                               uint3 domainHigh,
                               uint3 domainSize,
                               double* d_absk,
                               double* d_sigmaT4,
                               double* d_divQ,
                               double* d_VRFlux)
{

}  // end ray trace kernel


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
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
                                 double& sumI)
{

  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;
  int dx = domainLo.x;
  int dy = domainHi.y;

  int3 cur = origin;
  int3 prevCell = cur;

  // Step and sign for ray marching
  int step[3];                                          // Gives +1 or -1 based on sign
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

  double DyDxRatio = Dx.y / Dx.x;  //noncubic
  double DzDxRatio = Dx.z / Dx.x;  //noncubic

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

  //+++++++Begin ray tracing+++++++++++++++++++

  // Vector temp_direction = direction_vector;   // Used for reflections

  //save the direction vector so that it can get modified by...
  //the 2nd switch statement for reflections, but so that we can get the ray_location back into...
  //the domain after it was updated following the first switch statement.

  int nReflect = 0;  // Number of reflections that a ray has undergone
  //Threshold while loop
  while (intensity > threshold) {
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

      in_domain = containsCellKernel(domainLo, domainHi, cur, face);

      //__________________________________
      //  Update the ray location
      //this is necessary to find the absorb_coef at the endpoints of each step if doing interpolations
      //ray_location_prev = ray_location;
      //ray_location      = ray_location + (disMin * direction_vector);// If this line is used,  make sure that direction_vector is adjusted after a reflection

      // The running total of alpha*length
      double optical_thickness_prev = optical_thickness;
      //as long as tDeltaY,Z tMaxY,Z and ray_location[1],[2]..
      optical_thickness += Dx.x * abskg[INDEX3D(dx,dy,prevCell.x,prevCell.y,prevCell.z)] * disMin;
      // were adjusted by DyDxRatio or DzDxRatio, this line is now correct for noncubic domains.

      size++;

      //Eqn 3-15(see below reference) while
      //Third term inside the parentheses is accounted for in Inet. Chi is accounted for in Inet calc.
      sumI += sigmaT4Pi[INDEX3D(dx,dy,prevCell.x,prevCell.y,prevCell.z)] * (exp(-optical_thickness_prev) - exp(-optical_thickness)) * fs;

    }  //end domain while loop.  ++++++++++++++

    intensity = exp(-optical_thickness);

    //  wall emission 12/15/11
    sumI += abskg[INDEX3D(dx,dy,cur.x,cur.y,cur.z)] * sigmaT4Pi[INDEX3D(dx,dy,cur.x,cur.y,cur.z)] * intensity;

    intensity = intensity * (1 - abskg[INDEX3D(dx,dy,cur.x,cur.y,cur.z)]);

    //__________________________________
    //  Reflections
    if (intensity > threshold) {

      ++nReflect;
      fs = fs * (1 - abskg[INDEX3D(dx,dy,cur.x,cur.y,cur.z)]);

      //put cur back inside the domain
      cur = prevCell;

      // apply reflection condition
      step[face] *= -1;                      // begin stepping in opposite direction
      sign[face] = (sign[face] == 1) ? 0 : 1;  //  swap sign from 1 to 0 or vice versa

      in_domain = 1;

    }  // if reflection
  }  // threshold while loop.
}  // end of updateSumI function


//---------------------------------------------------------------------------
// Device Function:
//---------------------------------------------------------------------------
__device__ bool containsCellKernel(const int3& low,
                                   const int3& high,
                                   const int3& cell,
                                   const int&  face)
{
  switch (face) {
    case 0 :
      return low.x <= cell.x && high.x > cell.x;
    case 1 :
      return low.y <= cell.y && high.y > cell.y;
    case 2 :
      return low.z <= cell.z && high.z > cell.z;
    default :
      return false;
  }
}



