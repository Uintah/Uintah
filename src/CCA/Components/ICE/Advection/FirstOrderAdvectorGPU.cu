/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/ICE/Advection/FirstOrderAdvectorGPU.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
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
  __shared__ bool error;    // SHOULD THIS BE SET TO FALSE?
  
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

      //__________________________________
      //  Bullet proofing
      double total_fluxout = top + bottom + right + left + front + back;
  
      if(total_fluxout > (cellSizes.x*cellSizes.y*cellSizes.z)){
        error = true;
      }
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






/* ---------------------------------------------------------------------
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
 ---------------------------------------------------------------------  */
FirstOrderAdvectorGPU::FirstOrderAdvectorGPU() 
{
}


FirstOrderAdvectorGPU::FirstOrderAdvectorGPU(DataWarehouse* new_dw, 
                                       const Patch* patch,
                                       const bool isNewGrid)
{
  new_dw->allocateTemporary(d_OFS,patch, Ghost::AroundCells,1);
  
  // Initialize temporary variables when the grid changes
  if(isNewGrid){   
    double EVILNUM = -9.99666999e30;
    int NGC = 1;  // number of ghostCells
    for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
 
      const IntVector& c = *iter;
      for(int face = TOP; face <= BACK; face++ )  {
        d_OFS[c].d_fflux[face]= EVILNUM;
      }
    }
  }
}


FirstOrderAdvectorGPU::~FirstOrderAdvectorGPU()
{
}

FirstOrderAdvectorGPU* FirstOrderAdvectorGPU::clone(DataWarehouse* new_dw,
                                              const Patch* patch,
                                              const bool isNewGrid)
{
  return scinew FirstOrderAdvectorGPU(new_dw,patch,isNewGrid);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the slabs
 Steps for each cell:  
 1) calculate the volume for each outflux
 2) test to see if the total outflux > cell volume

Implementation notes:
The outflux of volume is calculated in each cell in the computational domain
+ one layer of extra cells  surrounding the domain.The face-centered velocity 
needs to be defined on all faces for these cells 

See schematic diagram at bottom of ice.cc for del* definitions
 ---------------------------------------------------------------------  */

void FirstOrderAdvectorGPU::inFluxOutFluxVolume(
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT, 
                        const Patch* patch,
                        const int& indx,
                        const bool& bulletProof_test,
                        DataWarehouse* new_dw)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells  
  bool error = false;
  int NGC = 1;  // number of ghostCells

  int num_devices, device;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 1) {
    int max_multiprocessors = 0, max_device = 0;
    for (device = 0; device < num_devices; device++) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      if (max_multiprocessors < properties.multiProcessorCount) {
        max_multiprocessors = properties.multiProcessorCount;
        max_device = device;
      }
    }
    cudaSetDevice(max_device);
  }


  IntVector l      = patch->getExtraCellLowIndex(NGC);
  IntVector h      = patch->getExtraCellHighIndex(NGC);
  IntVector s      = (h-l);
  int xdim = s.x(), ydim = s.y(), zdim = s.z();
  int size         = s.x()*s.y()*s.z()*sizeof(double);

  /*
  // device pointers
  double *uuvel_FC;
  double *vvvel_FC;
  double *wwvel_FC;
  double *OFS;
  const double *hostUvel_FC = uvel_FC.getWindow()->getData()->getPointer();
  const double *hostVvel_FC = vvel_FC.getWindow()->getData()->getPointer();
  const double *hostWvel_FC = wvel_FC.getWindow()->getData()->getPointer();
  const double *hostOFS     = (const double *)d_OFS.getWindow()->getData()->getPointer();

  // Memory copies host->device
  cudaMalloc(&uuvel_FC, size);
  cudaMalloc(&vvvel_FC, size);
  cudaMalloc(&wwvel_FC, size);
  cudaMalloc(&OFS,    6*size);

  cudaMemcpy(uuvel_FC, hostUvel_FC,
             size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(vvvel_FC, hostVvel_FC,
             size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(wwvel_FC, hostWvel_FC,
             size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(OFS, hostOFS,
             6*size,
             cudaMemcpyHostToDevice);


  // ADD PINNED FOR THE "error" VARIABLE!!!

  // set up domian specs
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
  if( xdim % 8 != 0)
  {
    xBlocks++;
  }
  int yBlocks = ydim / 8;
  if( ydim % 8 != 0)
  {
    yBlocks++;
  }
  dim3 totalBlocks(xBlocks,yBlocks);

  // Kernel invocation
  inFluxOutFluxVolumeKernel<<< totalBlocks, threadsPerBlock >>>(domainLower,
                                                                domainHigh,
                                                                domainSize,
                                                                cellSizes,
                                                                ghostLayers,
                                                                delT,
                                                                uuvel_FC,
                                                                vvvel_FC,
                                                                wwvel_FC,
                                                                OFS);
  cudaDeviceSynchronize();

  // Memory copies device->host
  cudaMemcpy((void *)hostUvel_FC, uuvel_FC,
             size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)hostVvel_FC, vvvel_FC,
             size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)hostWvel_FC, wwvel_FC,
             size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)hostOFS, OFS,
             6*size,
             cudaMemcpyDeviceToHost);


  // Free up memory
  cudaFree(uuvel_FC);
  cudaFree(vvvel_FC);
  cudaFree(wwvel_FC);
  cudaFree(OFS);

*/
  const double *uvel_FCD = uvel_FC.getWindow()->getData()->getPointer();
  const double *vvel_FCD = vvel_FC.getWindow()->getData()->getPointer();
  const double *wvel_FCD = wvel_FC.getWindow()->getData()->getPointer();
  double *OFS            = (double *)d_OFS.getWindow()->getData()->getPointer();


  double delt = delT;
  int tidX = 0;
  int tidY = 0;
  int num_slices = h.z();
  int d = h.x();
  int dy = h.y();
  int dxx = s.x();
  int dyy = s.y()-1;
  std::cout << "height: " << h << endl;
  std::cout << "dxx: " << dxx << endl;
  std::cout << "dyy: " << dyy << endl;
  std::cout << "slices: " << num_slices << endl;
  // In computing fluxes, we need the host cells too
  for (int slice = 0; slice <= num_slices; slice++)
  {
   for(tidY = 0; tidY <= dy; tidY++)
   {
     for(tidX = 0; tidX <= d; tidX++)
     {
      //std::cout << "x: " << tidX << "  y: " << tidY << "  z: " << slice << endl;
      std::cout << "[int " << tidX << ", " << tidY << ", " << slice <<"]" << endl;
      int index = INDEX3D(dxx,dyy, tidX,tidY, slice);
      //std::cout << index << endl;
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
      valueAdjacent = vvel_FCD[INDEX3D(dxx,dyy, tidX,(tidY + 1), slice)];
      delY_top      = std::max(0.0, valueAdjacent * delt);

      valueAdjacent = uvel_FCD[INDEX3D(dxx,dyy, (tidX + 1),tidY, slice)];
      delX_right    = std::max(0.0, valueAdjacent * delt);

      valueAdjacent = wvel_FCD[INDEX3D(dxx,dyy, tidX,tidY, (slice + 1))];
      delZ_front    = std::max(0.0, valueAdjacent * delt);



      // The minus
      valueAdjacent = vvel_FCD[index];
      delY_bottom   = std::max(0.0, -(valueAdjacent * delt));

      valueAdjacent = uvel_FCD[index];
      delX_left     = std::max(0.0, -(valueAdjacent * delt));

      valueAdjacent = wvel_FCD[index];
      delZ_back     = std::max(0.0, -(valueAdjacent * delt));

     //__________________________________
      //   SLAB outfluxes
      double delX_Z = dx.x() * dx.z();
      double delX_Y = dx.x() * dx.y();
      double delY_Z = dx.y() * dx.z();
      double top    = delY_top    * delX_Z;
      double bottom = delY_bottom * delX_Z;
      double right  = delX_right  * delY_Z;
      double left   = delX_left   * delY_Z;
      double front  = delZ_front  * delX_Y;
      double back   = delZ_back   * delX_Y;

      // copy values to correct values of OFS
      OFS[6*index + 0] = top;
      OFS[6*index+1] = bottom;
      OFS[6*index+2] = right;
      OFS[6*index+3] = left;
      OFS[6*index+4] = front;
      OFS[6*index+5] = back;

      //__________________________________
      //  Bullet proofing
      double total_fluxout = top + bottom + right + left + front + back;
      std::cout << "total_fluxout: " << total_fluxout << endl;
      if(total_fluxout > (dx.x()*dx.y()*dx.z())){
        error = true;
      }
     }
    }
  }
  // device and host memory pointers

  
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux slab vol in all cells = 0.0,
  // -request that the timestep be restarted.
  // -ignore if a timestep restart has already been requested
  bool tsr = new_dw->timestepRestarted();
  
  if (error && bulletProof_test && !tsr) {
    vector<IntVector> badCells;
    vector<fflux>  badOutflux;
    
    for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {
      IntVector c = *iter; 
      double total_fluxout = 0.0;
      fflux& ofs = d_OFS[c];
      
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += d_OFS[c].d_fflux[face];
        d_OFS[c].d_fflux[face] = 0.0;
      }
      // keep track of which cells are bad
      if (vol - total_fluxout < 0.0) {
        badCells.push_back(c);
        badOutflux.push_back(ofs);
      }
    }  // cell iter
    warning_restartTimestep( badCells,badOutflux, vol, indx, patch, new_dw);
  }  // if total_fluxout > vol
  
  if (error && !bulletProof_test) {
    std::ostringstream mesg;
    std::cout << " WARNING: ICE Advection operator Influx/Outflux volume error:"
         << " Patch " << patch->getID()
              << ", Level " << patch->getLevel()->getIndex()<< std::endl;
  }
}

/*_____________________________________________________________________
 Function~ advectQ
_____________________________________________________________________*/
//     M A S S
void FirstOrderAdvectorGPU::advectMass(const CCVariable<double>& q_CC,
                                    CCVariable<double>& q_advected,
                                    advectVarBasket* varBasket)
{
        
  advectSlabs<double>(q_CC,varBasket->patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignore_q_FC_calc_D());
                      
  // fluxes on faces at the coarse fine interfaces                    
  q_FC_fluxes<double>(q_CC, "mass", varBasket);                
}

//__________________________________
//     D O U B L E 
// (int_eng, sp_vol * mass, transported Variables)
void FirstOrderAdvectorGPU::advectQ(const CCVariable<double>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<double>& q_advected,
                                 advectVarBasket* varBasket)
{                                 
  advectSlabs<double>(q_CC,varBasket->patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignore_q_FC_calc_D());
                      
  // fluxes on faces at the coarse fine interfaces                    
  q_FC_fluxes<double>(q_CC, varBasket->desc, varBasket);
}

//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit ICE  q_CC = volFrac
void FirstOrderAdvectorGPU::advectQ(const CCVariable<double>& q_CC,
                                 const Patch* patch,
                                 CCVariable<double>& q_advected,
                                 advectVarBasket* varBasket,
                                 SFCXVariable<double>& q_XFC,
                                 SFCYVariable<double>& q_YFC,
                                 SFCZVariable<double>& q_ZFC,
                                     DataWarehouse* /*new_dw*/)
{
  advectSlabs<double>(q_CC,patch,q_advected,  
                      q_XFC, q_YFC, q_ZFC, save_q_FC());
                      
  // fluxes on faces at the coarse fine interfaces                    
  q_FC_PlusFaces( q_CC, patch, q_XFC, q_YFC, q_ZFC); 
  
  // fluxes on faces at the coarse fine interfaces                    
  q_FC_fluxes<double>(q_CC, "vol_frac", varBasket);
}
//__________________________________
//     V E C T O R  (momentum)
void FirstOrderAdvectorGPU::advectQ(const CCVariable<Vector>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<Vector>& q_advected,
                                 advectVarBasket* varBasket)
{
  advectSlabs<Vector>(q_CC,varBasket->patch,q_advected, 
                      d_notUsedX, d_notUsedY, d_notUsedZ, 
                      ignore_q_FC_calc_V());
                      
  // fluxes on faces at the coarse fine interfaces
  q_FC_fluxes<Vector>(q_CC, varBasket->desc, varBasket);
} 

/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T, typename F> 
  void FirstOrderAdvectorGPU::advectSlabs(const CCVariable<T>& q_CC,
                                       const Patch* patch,                   
                                       CCVariable<T>& q_advected,
                                       SFCXVariable<double>& q_XFC,
                                       SFCYVariable<double>& q_YFC,
                                       SFCZVariable<double>& q_ZFC,
                                       F save_q_FC) // function is passed in
{                  
  Vector dx = patch->dCell();            
  double invvol = 1.0/(dx.x() * dx.y() * dx.z());                     

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    const IntVector& c = *iter;  
    
    T q_face_flux[6];
    double faceVol[6];
    
    T sum_q_face_flux(0.0);   
    for(int f = TOP; f <= BACK; f++ )  {    
      //__________________________________
      //   S L A B S
      // q_CC: vol_frac, mass, momentum, int_eng....
      //      for consistent units you need to divide by cell volume
      // 
      IntVector ac = c + S_ac[f];     // slab adjacent cell
      double outfluxVol = d_OFS[c ].d_fflux[OF_slab[f]];
      double influxVol  = d_OFS[ac].d_fflux[IF_slab[f]];

      T q_faceFlux_tmp  =   q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
        
      faceVol[f]       =  outfluxVol +  influxVol;
      q_face_flux[f]   = q_faceFlux_tmp; 
      sum_q_face_flux += q_faceFlux_tmp;
    }  
    q_advected[c] = sum_q_face_flux*invvol;
    
    //__________________________________
    //  inline function to compute q_FC
    save_q_FC(c, q_XFC, q_YFC, q_ZFC, faceVol, q_face_flux, q_CC); 
  }
}
/*_____________________________________________________________________
 Function~ q_FC_operator
 Compute q at the face center.
_____________________________________________________________________*/
template<class T>
void FirstOrderAdvectorGPU::q_FC_operator(CellIterator iter, 
                                       IntVector adj_offset,
                                       const int face,
                                       const CCVariable<double>& q_CC,
                                       T& q_FC)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;      
    IntVector L = R + adj_offset; 

     // face:           LEFT,   BOTTOM,   BACK  
     // IF_slab[face]:  RIGHT,  TOP,      FRONT
    double outfluxVol = d_OFS[R].d_fflux[face];
    double influxVol  = d_OFS[L].d_fflux[IF_slab[face]];

    double q_faceFlux = q_CC[L] * influxVol - q_CC[R] * outfluxVol;
    double faceVol = outfluxVol + influxVol;

    double q_tmp_FC = fabs(q_faceFlux)/(faceVol + 1.0e-100);

    // if q_tmp_FC = 0.0 then set it equal to q_CC[c]
    q_FC[R] = equalZero(q_faceFlux, q_CC[R], q_tmp_FC);
  }
}

/*_____________________________________________________________________
 Function~  q_FC_PlusFaces
 Compute q_FC values on the faces between the extra cells
 and the interior domain only on the x+, y+, z+ patch faces 
_____________________________________________________________________*/
void FirstOrderAdvectorGPU::q_FC_PlusFaces(
                                       const CCVariable<double>& q_CC,
                                   const Patch* patch,
                                   SFCXVariable<double>& q_XFC,
                                   SFCYVariable<double>& q_YFC,
                                   SFCZVariable<double>& q_ZFC)
{                                                  
  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);    // X faces
  adj_offset[1] = IntVector(0, -1, 0);    // Y faces
  adj_offset[2] = IntVector(0,  0, -1);   // Z faces
  
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  
  IntVector patchOnBoundary = patch->noNeighborsHigh();
  // only work on patches that are at the edge of the computational domain
  
  if (patchOnBoundary.x() == 1 ){
    CellIterator Xiter=patch->getFaceIterator(Patch::xplus,MEC);
    q_FC_operator<SFCXVariable<double> >(Xiter, adj_offset[0], LEFT,  
                                         q_CC,q_XFC);
  } 
  if (patchOnBoundary.y() == 1 ){
    CellIterator Yiter=patch->getFaceIterator(Patch::yplus,MEC);
    q_FC_operator<SFCYVariable<double> >(Yiter, adj_offset[1], BOTTOM,
                                         q_CC,q_YFC); 
  }
  if (patchOnBoundary.z() == 1 ){  
    CellIterator Ziter=patch->getFaceIterator(Patch::zplus,MEC);
    q_FC_operator<SFCZVariable<double> >(Ziter, adj_offset[2], BACK,  
                                         q_CC,q_ZFC);  
  }
}
/*_____________________________________________________________________
 Function~ q_FC_flux_operator
 Compute the flux of q across a face.  The flux is need by the AMR 
 refluxing operation
_____________________________________________________________________*/
template<class T, class V>
void FirstOrderAdvectorGPU::q_FC_flux_operator(CellIterator iter, 
                                          IntVector adj_offset,
                                          const int face,
                                          const CCVariable<V>& q_CC,
                                          T& q_FC_flux)
{
  int out_indx = OF_slab[face]; //LEFT,   BOTTOM,   BACK 
  int in_indx  = IF_slab[face]; //RIGHT,  TOP,      FRONT

  for(;!iter.done(); iter++){
    IntVector c = *iter;      
    IntVector ac = c + adj_offset; 

    double outfluxVol = d_OFS[c].d_fflux[out_indx];
    double influxVol  = d_OFS[ac].d_fflux[in_indx];

    q_FC_flux[c] += q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
    
  }  
}
/*_____________________________________________________________________
 Function~  q_FC_fluxes
 Computes the sum(flux of q at the face center) over all subcycle timesteps
 on the fine level.  We only *need* to hit the cell that are on a coarse-fine 
 interface, ignoring the extraCells.  However, this routine computes 
 the fluxes over the entire computational domain, which could be slow.
 Version r29970 has the fluxes computed on the fine level at the coarse
 fine interfaces.  You need to add the same computation on the coarse 
 level. Note that on the coarse level you don't know where the coarse fine
 interfaces are and need to look up one level to find the interfaces.
_____________________________________________________________________*/
template<class T>
void FirstOrderAdvectorGPU::q_FC_fluxes( const CCVariable<T>& q_CC,
                                      const string& desc,
                                      advectVarBasket* vb)
{
  if(vb->doRefluxing){
    // pull variables from the basket
    const int indx = vb->indx;
    const Patch* patch = vb->patch;
    DataWarehouse* new_dw = vb->new_dw;
    DataWarehouse* old_dw = vb->old_dw;
    const double AMR_subCycleProgressVar = vb->AMR_subCycleProgressVar;

    // form the label names
    string x_name = desc + "_X_FC_flux";
    string y_name = desc + "_Y_FC_flux";
    string z_name = desc + "_Z_FC_flux";
    // get the varLabels
    VarLabel* xlabel = VarLabel::find(x_name);
    VarLabel* ylabel = VarLabel::find(y_name);
    VarLabel* zlabel = VarLabel::find(z_name);  
    if (xlabel == NULL || ylabel == NULL || zlabel == NULL){
      throw InternalError( "Advector: q_FC_fluxes: variable label not found: " 
                            + x_name + " or " + y_name + " or " + z_name, __FILE__, __LINE__);
    }
    Ghost::GhostType  gn  = Ghost::None;
    SFCXVariable<T> q_X_FC_flux;
    SFCYVariable<T> q_Y_FC_flux;
    SFCZVariable<T> q_Z_FC_flux;

    new_dw->allocateAndPut(q_X_FC_flux, xlabel,indx, patch);
    new_dw->allocateAndPut(q_Y_FC_flux, ylabel,indx, patch);
    new_dw->allocateAndPut(q_Z_FC_flux, zlabel,indx, patch); 

    if(AMR_subCycleProgressVar == 0){
      q_X_FC_flux.initialize(T(0.0));   // at the beginning of the cycle 
      q_Y_FC_flux.initialize(T(0.0));   // initialize the fluxes
      q_Z_FC_flux.initialize(T(0.0));
    }else{
      constSFCXVariable<T> q_X_FC_flux_old;
      constSFCYVariable<T> q_Y_FC_flux_old;
      constSFCZVariable<T> q_Z_FC_flux_old;

      old_dw->get(q_X_FC_flux_old, xlabel, indx, patch, gn,0);
      old_dw->get(q_Y_FC_flux_old, ylabel, indx, patch, gn,0);
      old_dw->get(q_Z_FC_flux_old, zlabel, indx, patch, gn,0);

      q_X_FC_flux.copyData(q_X_FC_flux_old);
      q_Y_FC_flux.copyData(q_Y_FC_flux_old);
      q_Z_FC_flux.copyData(q_Z_FC_flux_old);
    }
    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces
                       
    CellIterator XFC_iter = patch->getSFCXIterator();
    CellIterator YFC_iter = patch->getSFCYIterator();
    CellIterator ZFC_iter = patch->getSFCZIterator();
    
    q_FC_flux_operator<SFCXVariable<T>, T>(XFC_iter, adj_offset[0],LEFT,
                                           q_CC,q_X_FC_flux); 

    q_FC_flux_operator<SFCYVariable<T>, T>(YFC_iter, adj_offset[1],BOTTOM,
                                           q_CC,q_Y_FC_flux); 

    q_FC_flux_operator<SFCZVariable<T>, T>(ZFC_iter, adj_offset[2],BACK,
                                           q_CC,q_Z_FC_flux);
                                           
 /*`==========TESTING==========*/    
#ifdef SPEW                
    vector<Patch::FaceType> cf;
    patch->getCoarseFaces(cf);
    vector<Patch::FaceType>::const_iterator itr;  
    for (itr = cf.begin(); itr != cf.end(); ++itr){
      Patch::FaceType patchFace = *itr;
      string name = patch->getFaceName(patchFace);


      if(is_rightFace_variable(name,desc)){
          cout << " ------------ FirstOrderAdvectorGPU::q_FC_fluxes " << desc<< endl;
        cout << "AMR_subCycleProgressVar " << AMR_subCycleProgressVar << " Level " << patch->getLevel()->getIndex()
              << " Patch " << patch->getGridIndex()<< endl;
        cout <<" patchFace " << name << " " ;

        IntVector shift = patch->faceDirection(patchFace);
        shift = SCIRun::Max(IntVector(0,0,0), shift);  // set -1 values to 0

        Patch::FaceIteratorType IFC = Patch::InteriorFaceCells;

        CellIterator iter =patch->getFaceIterator(patchFace, IFC);
        IntVector begin = iter.begin() + shift;
        IntVector end   = iter.end() + shift;

        IntVector half  = (end - begin)/IntVector(2,2,2) + begin;
        if(patchFace == Patch::xminus || patchFace == Patch::xplus){
          cout << half << " \t sum_q_flux " << q_X_FC_flux[half] <<  endl; 
        } 
        if(patchFace == Patch::yminus || patchFace == Patch::yplus){
          cout << half << " \t sum_q_flux " << q_Y_FC_flux[half] <<  endl;
        }
        if(patchFace == Patch::zminus || patchFace == Patch::zplus){
          cout << half << " \t sum_q_flux " << q_Z_FC_flux[half] <<  endl;
        }
      } 
    } 
#endif
  /*===========TESTING==========`*/                                       
                                           
  } // doRefluxing   
}





