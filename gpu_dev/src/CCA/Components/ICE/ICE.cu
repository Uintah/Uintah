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
#include <stdio.h>
#include <sci_defs/cuda_defs.h>
#include <CCA/Components/ICE/GpuModule.h>
#include <Core/Grid/Variables/GPUGridVariable.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
//no linker for device code, need to include the whole source...
#include <CCA/Components/Schedulers/GPUDataWarehouse.cu>
#include <Core/Grid/Variables/GPUGridVariable.cu>
#ifdef __cplusplus
extern "C" {
#endif

namespace Uintah {

__device__ void idealGas_ComputePressEOS(double& rhoM, double& gamma,
                            double& cv, double& Temp,
                            double& press, double& dp_drho, double& dp_de){
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;

}

__device__ double idealGas_ComputeRhoMicro(double& press, double& gamma, double& cv, double& Temp, double) {
  return  press/((gamma - 1.0)*cv*Temp);
  
}

__global__ void IceEquilibrationKernelUnified(
                          double d_SMALL_NUM,
                          int d_max_iter_equilibration,
                          double convergence_crit,
                          int patchID,
                          uint3 size,
                          int zSliceThickness,
                          GPUDataWarehouse *old_gpudw,
                          GPUDataWarehouse *new_gpudw) {


    //temporary variables
  /*
  GPUGridVariable<double> gpu_press_eos;
  new_gpudw->get(gpu_press_eos, "press_eos", patchID, 0);
  GPUGridVariable<double> gpu_dp_drho;
  new_gpudw->get(gpu_dp_drho, "dp_drho", patchID, 0);
  GPUGridVariable<double> gpu_dp_de;
  new_gpudw->get(gpu_dp_de, "dp_de", patchID, 0);
  GPUGridVariable<double> gpu_rho_micro;
  new_gpudw->get(gpu_rho_micro, "rho_micro", patchID, 0);
  GPUGridVariable<double> gpu_vol_frac;
  new_gpudw->get(gpu_vol_frac, "vol_frac_CC", patchID, 0);

  GPUGridVariable<double> gpu_speedSound_new;
  new_gpudw->get(gpu_speedSound_new, "speedSound_CC", patchID, 0);
  GPUGridVariable<double> gpu_Temp;
  old_gpudw->get(gpu_Temp, "temp_CC", patchID, 0);
  GPUGridVariable<double> gpu_rho_CC;
  old_gpudw->get(gpu_rho_CC, "rho_CC", patchID, 0);
  GPUGridVariable<double> gpu_sp_vol_CC;
  old_gpudw->get(gpu_sp_vol_CC, "sp_vol_CC", patchID, 0);
  GPUGridVariable<double> gpu_cv;
  new_gpudw->get(gpu_cv, "specific_heat", patchID, 0);
  GPUGridVariable<double> gpu_gamma;
  new_gpudw->get(gpu_gamma, "gamma", patchID, 0); */

  //Get the data.  Do it here so all threads are involved searching for the
  //data addresses
  __shared__ GPUGridVariable<double> gpu_press_CC;
  old_gpudw->get(gpu_press_CC, "press_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_sp_vol_CC;
  old_gpudw->get(gpu_sp_vol_CC, "sp_vol_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_rho_CC;
  old_gpudw->get(gpu_rho_CC, "rho_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_Temp;
  old_gpudw->get(gpu_Temp, "temp_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_press_eos;
  new_gpudw->get(gpu_press_eos, "press_eos_temp", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_dp_drho;
  new_gpudw->get(gpu_dp_drho, "dp_drho_temp", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_dp_de;
  new_gpudw->get(gpu_dp_de, "dp_de_temp", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_press_new;
  new_gpudw->get(gpu_press_new, "press_equil_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_rho_micro;
  new_gpudw->get(gpu_rho_micro, "rho_micro_temp", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_vol_frac;
    new_gpudw->get(gpu_vol_frac, "vol_frac_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_speedSound_new;
  new_gpudw->get(gpu_speedSound_new, "speedSound_CC", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_cv;
  new_gpudw->get(gpu_cv, "specific_heat", patchID, 0);
  __shared__ GPUGridVariable<double> gpu_gamma;
  new_gpudw->get(gpu_gamma, "gamma", patchID, 0);



  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int numMaterials = new_gpudw->getNumMaterials();

  //Only assign one thread per cell in the z-slice.  Some threads may not have cells to point to.
  if ((i < size.x) && (j < size.y)) {
    int startingZSlice = blockIdx.z * zSliceThickness;
    int endingZSlice = startingZSlice + zSliceThickness;
    if (endingZSlice > size.z) {
      endingZSlice = size.z;
    }
    //go down the z slices
    for (int k = startingZSlice; k < endingZSlice; k++ ) {

      //copy manually
      gpu_press_new(i,j,k) = gpu_press_CC(i,j,k);

      double delPress = 0.0;
      double sum = 0.0;
      double tmp;
      int count = 0;

      __shared__ bool converged[32][32]; //can't allow for more threads in a block than this.

      converged[threadIdx.x][threadIdx.y] = false;

      for (int m = 0; m < numMaterials; m++) {
          //printf("For %d %d %d %d I have value %1.16e\n", i,j,k,m, gpu_sp_vol_CC(i,j,k,m));
          gpu_rho_micro(i,j,k,m) = 1.0 / gpu_sp_vol_CC(i,j,k,m);
          gpu_vol_frac(i,j,k,m) = gpu_rho_CC(i,j,k,m) * gpu_sp_vol_CC(i,j,k,m);
      }
      while (count < d_max_iter_equilibration && converged[threadIdx.x][threadIdx.y]  == false) {
        count++;

        for (int m = 0; m < numMaterials; m++) {
          if (new_gpudw->getMaterial(m) == IDEAL_GAS) {
            idealGas_ComputePressEOS(gpu_rho_micro(i,j,k,m),
                       gpu_gamma(i,j,k,m),
                       gpu_cv(i,j,k,m),
                       gpu_Temp(i,j,k,m),
                       gpu_press_eos(i,j,k,m),
                       gpu_dp_drho(i,j,k,m),
                       gpu_dp_de(i,j,k,m));
          }
        }
        double A = 0.0, B = 0.0, C = 0.0;

        for (int m = 0; m < numMaterials; m++) {

          double Q = gpu_press_new(i,j,k) - gpu_press_eos(i,j,k,m);

          double div_y = (gpu_vol_frac(i,j,k,m) * gpu_vol_frac(i,j,k,m))
                  / (gpu_dp_drho(i,j,k,m) * gpu_rho_CC(i,j,k,m) + d_SMALL_NUM);
          A += gpu_vol_frac(i,j,k,m);
          B += Q * div_y;
          C += div_y;


        }

        double vol_frac_not_close_packed = 1.0;
        delPress = (A - vol_frac_not_close_packed - B) / C;
        gpu_press_new(i,j,k) += delPress;


        for (int m = 0; m < numMaterials; m++) {

          if (new_gpudw->getMaterial(m) == IDEAL_GAS) {
            gpu_rho_micro(i,j,k,m) = idealGas_ComputeRhoMicro(
                             gpu_press_new(i,j,k),
                             gpu_gamma(i,j,k,m),
                             gpu_cv(i,j,k,m),
                             gpu_Temp(i,j,k,m),
                             gpu_rho_micro(i,j,k,m));
          }

          double div = 1. / gpu_rho_micro(i,j,k,m);
          // - updated volume fractions
          gpu_vol_frac(i,j,k,m) = gpu_rho_CC(i,j,k,m) * div;

        }

        //__________________________________
        // - Test for convergence
        //  If sum of vol_frac_CC ~= vol_frac_not_close_packed then converged
        sum = 0.0;
        for (int m = 0; m < numMaterials; m++) {
          sum += gpu_vol_frac(i,j,k,m);
        }
        if (fabs(sum - 1.0) < convergence_crit) {
          converged[threadIdx.x][threadIdx.y]  = true;
          // Find the speed of sound based on converged solution
          for (int m = 0; m < numMaterials; m++) {
            if (new_gpudw->getMaterial(m) == IDEAL_GAS) {
              idealGas_ComputePressEOS(gpu_rho_micro(i,j,k,m),
                         gpu_gamma(i,j,k,m),
                         gpu_cv(i,j,k,m),
                         gpu_Temp(i,j,k,m),
                         gpu_press_eos(i,j,k,m),
                         gpu_dp_drho(i,j,k,m),
                         gpu_dp_de(i,j,k,m));
            }

            tmp = gpu_dp_drho(i,j,k,m) +
                  gpu_dp_de(i,j,k,m) *
                  gpu_press_eos(i,j,k,m) /
                  (gpu_rho_micro(i,j,k,m) * gpu_rho_micro(i,j,k,m));

            gpu_speedSound_new(i,j,k,m) = sqrt(tmp);
          }
        }
      }
    }
  }
}

void launchIceEquilibrationKernelUnified(dim3 dimGrid,
                          dim3 dimBlock,
                          cudaStream_t* stream,
                          uint3 size,
                          double d_SMALL_NUM,
                          int d_max_iter_equilibration,
                          double convergence_crit,
                          int patchID,
                          int zSliceThickness,
                          GPUDataWarehouse* old_gpudw,
                          GPUDataWarehouse* new_gpudw) {

  IceEquilibrationKernelUnified<<< dimGrid, dimBlock, 0, *stream  >>>(
  //IceEquilibrationKernelUnified<<< dimGrid, dimBlock, 0 >>>(
                          d_SMALL_NUM,
                          d_max_iter_equilibration,
                          convergence_crit,
                          patchID,
                          size,
                          zSliceThickness,
                          old_gpudw,
                          new_gpudw);

  cudaError_t cudaResult;
  cudaResult = cudaGetLastError();
  if (cudaResult != cudaSuccess)
  {
     printf("Error loading IceEquilibrationKernelUnified, error is: %d\n", cudaResult);
  }
  //cudaDeviceSynchronize();



}
#ifdef __cplusplus
}
#endif

} //end namespace Uintah


