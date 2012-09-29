/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/Advection/FirstOrderAdvector.h>
#include <Core/Geometry/IntVector.h>
#include <iostream>

//#define SPEW
#undef SPEW
#define is_rightFace_variable(face,var) ( ((face == "xminus" || face == "xplus") && var == "scalar-f") ?1:0  )

using namespace Uintah;
using std::cerr;
using std::endl;


///////////////////////////
// GPU Kernel Prototypes //
///////////////////////////
// The kernel that computes influx and outflux values essentially replacing the cell iterator.
__global__ void inFluxOutFluxVolumeKernel(uint3 domainLow,
                                          uint3 domainHigh,
                                          uint3 domainSize,
                                          uint3 cellSizes,
                                          int ghostLayers,
                                          double delt,
                                          double *uvel_FC, 
                                          double *vvel_FC, 
                                          double *wvel_FC,
                                          double *OFS)
{
  // Compute the index
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

  int num_slices = domainHigh.z;
  int dx = domainSize.x;
  int dy = domainSize.y;
  
  // In computing fluxes, we need the host cells too
  if (tidX > 0 && tidY > 0 && tidX < domainHigh.x && tidY < domainHigh.y) {
    for (int slice = domainLow.z; slice < num_slices; slice++) {
      int index = INDEX3D(dx,dy, tidX,tidY, slice);
      double valueAdjacent = 0.0; // A temporary storage for adjacent values
      
      // Set to initial values, these are set are used as the else case
      //   for the following if statements.
      double delY_top    = 0.0;
      double delY_bottom = 0.0;
      double delX_right  = 0.0;
      double delX_left   = 0.0;
      double delZ_front  = 0.0;
      double delZ_back   = 0.0;
  
      // NOTE REFACTOR THIS SECTION TO USE fmaxf(x,y)
      // The plus
      valueAdjacent = vvel_FC[INDEX3D(dx,dy, tidX,tidY + 1, slice)];
      delY_top      = fmaxf(0.0, valueAdjacent * delt);
              
      valueAdjacent = uvel_FC[INDEX3D(dx,dy, tidX + 1,tidY, slice)];  
      delX_right    = fmaxf(0.0, valueAdjacent * delt);
        
      valueAdjacent = wvel_FC[INDEX3D(dx,dy, tidX,tidY, slice + 1)];   
      delZ_front    = fmaxf(0.0, valueAdjacent * delt);
    
    
      // The minus
      valueAdjacent = vvel_FC[index];
      delY_bottom   = fmaxf(0.0, -(valueAdjacent * delt));
        
      valueAdjacent = uvel_FC[index];
      delX_left     = fmaxf(0.0, -(valueAdjacent * delt));
        
      valueAdjacent = wvel_FC[index];
      delZ_back     = fmaxf(0.0, -(valueAdjacent * delt));
    
    
    
      //__________________________________
      //   SLAB outfluxes
      double delX_Z = cellSizes.x * cellSizes.z;
      double delX_Y = cellSizes.x * cellSizes.y;
      double delY_Z = cellSizes.y * cellSizes.z;
      double top    = delY_top    * delX_Z;
      double bottom = delY_bottom * delX_Z;
      double right  = delX_right  * delY_Z;
      double left   = delX_left   * delY_Z;
      double front  = delZ_front  * delX_Y;
      double back   = delZ_back   * delX_Y;
  
      // copy values to correct values of OFS
      OFS[index*6] = top;
      OFS[index*6+1] = bottom;
      OFS[index*6+2] = right;
      OFS[index*6+3] = left;
      OFS[index*6+4] = front;
      OFS[index*6+5] = back;
    }
  }
}

// A kernel that applies the advection operation to a number of slabs.
__global__ void advectSlabsKernel(uint3 domainLow,
                                  uint3 domainHigh,
                                  uint3 domainSize,
                                  int ghostLayers,
                                  double *q_CC,
                                  double *q_advected,
                                  double *q_XFC,
                                  double *q_YFC,
                                  double *q_ZFC,
                                  double **OFS,
                                  double invol)
{

  // calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

  int num_slices = domainHigh.z;;
  int dx = domainSize.x;
  int dy = domainSize.y;

  double q_face_flux[6];
  double faceVol[6];

  if (tidX < domainHigh.x && tidY < domainHigh.y && tidX > 0 && tidY > 0) {
    for (int slice = domainLow.z; slice < num_slices; slice++) {
      // Variables needed for each cell
      int cell = INDEX3D(dx,dy, tidX,tidY, slice);
      int adjCell;
      double sum_q_face_flux = 0.0;
      double outfluxVol;
      double influxVol;
      double q_faceFlux_tmp;

      // Unrolled 'for' loop Above
      adjCell          = INDEX3D(dx,dy, tidX, (tidY+1), slice);
      influxVol        = OFS[adjCell][0];
      outfluxVol       = OFS[cell][0];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[0]   = q_faceFlux_tmp;
      faceVol[0]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[0];

      // Below
      adjCell          = INDEX3D(dx,dy, tidX, (tidY-1), slice);
      influxVol        = OFS[adjCell][1];
      outfluxVol       = OFS[cell][1];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[1]   = q_faceFlux_tmp;
      faceVol[1]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[1];

      // Right
      adjCell          = INDEX3D(dx,dy, (tidX+1), tidY, slice);
      influxVol        = OFS[adjCell][2];
      outfluxVol       = OFS[cell][2];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[2]   = q_faceFlux_tmp;
      faceVol[2]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[2];

      // Left
      adjCell          = INDEX3D(dx,dy, (tidX-1), tidY, slice);
      influxVol        = OFS[adjCell][3];
      outfluxVol       = OFS[cell][3];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[3]   = q_faceFlux_tmp;
      faceVol[3]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[3];

      // Front
      adjCell          = INDEX3D(dx,dy, tidX, tidY, (slice-1));
      influxVol        = OFS[adjCell][4];
      outfluxVol       = OFS[cell][4];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[4]   = q_faceFlux_tmp;
      faceVol[4]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[4];

      // Back
      adjCell          = INDEX3D(dx,dy, tidX, tidY, (slice+1));
      influxVol        = OFS[adjCell][5];
      outfluxVol       = OFS[cell][5];
      q_faceFlux_tmp   = q_CC[adjCell]*influxVol - q_CC[cell]*outfluxVol;
      q_face_flux[5]   = q_faceFlux_tmp;
      faceVol[5]       = outfluxVol + influxVol;
      sum_q_face_flux += q_face_flux[5];

      // Sum all the Advected double
      q_advected[cell]     = sum_q_face_flux*invol;
      
      
      
      // This is equivalent to save_q_FC //
      double tempFC;    // for holding the temporary value used in the face center saving
      double q_tmp = q_CC[cell];
      
      // Note: BOTTOM = 1, LEFT = 3, BACK = 5
      // X
      tempFC = fabsf(q_face_flux[3]/(faceVol[3]+1e-100));
      q_XFC[cell] = (q_face_flux[3] == 0.0 ? q_tmp:tempFC);
      
      // Y
      tempFC = fabsf(q_face_flux[1]/(faceVol[1]+1e-100));
      q_YFC[cell] = (q_face_flux[1] == 0.0 ? q_tmp:tempFC);   
      
      // Z
      tempFC = fabsf(q_face_flux[5]/(faceVol[5]+1e-100));
      q_ZFC[cell] = (q_face_flux[5] == 0.0 ? q_tmp:tempFC);     
      
    }
  }
}

// A kernel that computes the total flux through a face.
__global__ void q_FC_operatorKernel(uint3 domainLow,
                                    uint3 domainHigh,
                                    uint3 domainSize,
                                    uint3 adjOffset,
                                    int ghostLayers,
                                    int face,
                                    int oppositeFace,
                                    double **OFS,
                                    double *q_CC,
                                    double *q_FC)
{
  // calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

  int num_slices = domainHigh.z;
  int dx = domainSize.x;
  int dy = domainSize.y;


   if (tidX < domainHigh.x && tidY < domainHigh.y && tidX > 0 && tidY > 0) {
    for (int slice = domainLow.x; slice < num_slices; slice++) {
      int index = INDEX3D(dx,dy, tidX, tidY, slice);
      int adjIndex = INDEX3D(dx,dy, tidX+adjOffset.x, tidY+adjOffset.y, slice+ adjOffset.z);
      
      double outfluxVol = OFS[index][face];
      double influxVol  = OFS[adjIndex][oppositeFace];
      double q_faceFlux = q_CC[index] * influxVol - q_CC[adjIndex] * outfluxVol;
      
      double q_tmp_FC = fabs(q_faceFlux)/((outfluxVol+influxVol) + 1.0e-100);

      if(q_faceFlux == 0.0)
        q_FC[adjIndex] = q_CC[adjIndex];
      else
        q_FC[adjIndex] = q_tmp_FC;
    }
  }
}
    
// A kernel that computes the flux of q across a face.  The flux is need by the AMR refluxing operation.
__global__ void q_FC_flux_operatorKernel(uint3 domainLow,
                                         uint3 domainHigh,
                                         uint3 domainSize,
                                         uint3 adjOffset,
                                         int ghostLayers,
                                         int face,
                                         int oppositeFace,
                                         double **OFS,
                                         double *q_CC,
                                         double *q_FC_flux)
{
  // calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

  int num_slices = domainHigh.z;
  int dx = domainSize.x;
  int dy = domainSize.y;

  if (tidX < domainHigh.x && tidY < domainHigh.y && tidX > 0 && tidY > 0) {
    for (int slice = domainLow.z; slice < num_slices; slice++) {
      int index = INDEX3D(dx,dy, tidX, tidY, slice);
      int adjIndex = INDEX3D(dx,dy, tidX+adjOffset.x, tidY+adjOffset.y, slice+ adjOffset.z);

      q_FC_flux[index] += q_CC[adjIndex]*OFS[index][face] - q_CC[index]*OFS[adjIndex][oppositeFace];
    }
  }
}


void FirstOrderAdvector::inFluxOutFluxVolumeGPU(const VarLabel* uvel_FCMELabel,
                                                const VarLabel* vvel_FCMELabel,
                                                const VarLabel* wvel_FCMELabel,
                                                const double& delT,
                                                const Patch* patch,
                                                const int& indx,
                                                const bool& bulletProof_test,
                                                DataWarehouse* new_dw,
                                                const int& device,
                                                GPUThreadedMPIScheduler* sched)
{
  cudaError_t retVal;
  CUDA_RT_SAFE_CALL( cudaSetDevice(device) );

  Vector dx = patch->dCell();

  //__________________________________
  //  At patch boundaries you need to extend the computational footprint by one cell in ghostCells
  const int NGC = 1;  // number of ghostCells

  const IntVector l = patch->getExtraCellLowIndex(NGC);
  const IntVector h = patch->getExtraCellHighIndex(NGC);
  const IntVector s = (h - l);
  const int xdim = s.x();
  const int ydim = s.y();

  // device pointers
  double *uvel_FC = sched->getDeviceRequiresPtr(uvel_FCMELabel, indx, patch);
  double *vvel_FC = sched->getDeviceRequiresPtr(vvel_FCMELabel, indx, patch);
  double *wvel_FC = sched->getDeviceRequiresPtr(wvel_FCMELabel, indx, patch);
  double *OFS = (double *)d_OFS.getWindow()->getData()->getPointer();

  // set up domain specs
  uint3 domainSize  = make_uint3(s.x(),  s.y(),  s.z());
  uint3 domainLower = make_uint3(l.x(),  l.y(),  l.z());
  uint3 domainHigh  = make_uint3(h.x(),  h.y(),  h.z());
  uint3 cellSizes   = make_uint3(dx.x(), dx.y(), dx.z());
  int ghostLayers   = 1;  // default for FirstOrderAdvector

  // Threads per block must be power of 2 in each direction.  Here
  //  8 is chosen as a test value in the x and y and 1 in the z,
  //  as each of these (x,y) threads streams through the z direction.
  dim3 threadsPerBlock(8, 8, 1);

  // Set up the number of blocks of threads in each direction accounting for any
  //  non-power of 8 end pieces.
  int xBlocks = xdim / 8;
  if( xdim % 8 != 0) { xBlocks++; }

  int yBlocks = ydim / 8;
  if( ydim % 8 != 0) { yBlocks++; }
  dim3 totalBlocks(xBlocks,yBlocks);

  // Kernel invocation
  cudaStream_t* stream = sched->getCudaStream(device);
  inFluxOutFluxVolumeKernel<<< totalBlocks, threadsPerBlock, 0, *stream >>>(domainLower,
                                                                            domainHigh,
                                                                            domainSize,
                                                                            cellSizes,
                                                                            ghostLayers,
                                                                            delT,
                                                                            uvel_FC,
                                                                            vvel_FC,
                                                                            wvel_FC,
                                                                            OFS);

  cudaEvent_t* uvelEvent = sched->getCudaEvent(device);
  cudaStream_t* uvelStream = sched->getCudaStream(device);
  sched->requestD2HCopy(uvel_FCMELabel, indx, patch, uvelStream, uvelEvent);

  cudaEvent_t* vvelEvent = sched->getCudaEvent(device);
  cudaStream_t* vvelStream = sched->getCudaStream(device);
  sched->requestD2HCopy(vvel_FCMELabel, indx, patch, vvelStream, vvelEvent);

  cudaEvent_t* wvelEvent = sched->getCudaEvent(device);
  cudaStream_t* wvelStream = sched->getCudaStream(device);
  sched->requestD2HCopy(wvel_FCMELabel, indx, patch, wvelStream, wvelEvent);

  // Kernel error checking (for now)
  retVal = cudaGetLastError();
  if (retVal != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
    exit(-1);
  }

  cudaStreamSynchronize(*stream);

}

