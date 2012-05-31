/*

 The MIT License

 Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and
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

#include <CCA/Components/Examples/PoissonGPU1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

#include <sci_defs/cuda_defs.h>

using namespace std;
using namespace Uintah;

PoissonGPU1::PoissonGPU1(const ProcessorGroup* myworld) : UintahParallelComponent(myworld) {

  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

PoissonGPU1::~PoissonGPU1() {
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void PoissonGPU1::problemSetup(const ProblemSpecP& params,
                               const ProblemSpecP& restart_prob_spec,
                               GridP& /*grid*/,
                               SimulationStateP& sharedState) {
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");

  poisson->require("delt", delt_);

  mymat_ = scinew
  SimpleMaterial();

  sharedState->registerSimpleMaterial(mymat_);
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleInitialize(const LevelP& level,
                                     SchedulerP& sched) {
  Task * task = scinew Task("PoissonGPU1::initialize", this, &PoissonGPU1::initialize);

  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleComputeStableTimestep(const LevelP& level,
                                                SchedulerP& sched) {
  Task * task = scinew Task("PoissonGPU1::computeStableTimestep", this, &PoissonGPU1::computeStableTimestep);

  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::scheduleTimeAdvance(const LevelP& level,
                                      SchedulerP& sched) {
  Task * task = scinew Task("PoissonGPU1::timeAdvanceGPU", this, &PoissonGPU1::timeAdvanceGPU);

  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void PoissonGPU1::computeStableTimestep(const ProcessorGroup* pg,
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw) {

  if (pg->myrank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}
//______________________________________________________________________
//
void PoissonGPU1::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* /*old_dw*/,
                             DataWarehouse* new_dw) {
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.0);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face = Patch::nextFace(face)) {
      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);
          const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*>(bcb);
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++) {
            phi[*nbound_ptr] = value;
          }
          delete bcb;
        }
      }
    }
    new_dw->put(sum_vartype(-1), residual_label);
  }
}

//______________________________________________________________________
//
void PoissonGPU1::timeAdvanceCPU(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  int matl = 0;
  int ghostLayers = 1;

  // Do time steps
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;

    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);
    NCVariable<double> newphi;

    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual=0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor?0:1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor?0:1);

    //__________________________________
    //  Stencil
    for(NodeIterator iter(l, h);!iter.done(); iter++){
      IntVector n = *iter;

      newphi[n]=(1./6)*(
        phi[n+IntVector(1,0,0)] + phi[n+IntVector(-1,0,0)] +
        phi[n+IntVector(0,1,0)] + phi[n+IntVector(0,-1,0)] +
        phi[n+IntVector(0,0,1)] + phi[n+IntVector(0,0,-1)]);

      double diff = newphi[n] - phi[n];
      residual += diff * diff;
    }
    new_dw->put(sum_vartype(residual), residual_label);
  } // end patch for loop
}

//______________________________________________________________________
//
void PoissonGPU1::timeAdvance1DP(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw) {

  int matl = 0;
  int ghostLayers = 1;

  // Do time steps
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector s = h - l;

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    //__________________________________
    // 1D-Pointer Stencil
    double *phi_data = (double*)phi.getWindow()->getData()->getPointer();
    double *newphi_data = (double*)newphi.getWindow()->getData()->getPointer();

    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();
    int ghostLayers = 1;
    int ystride = yhigh + ghostLayers;
    int xstride = xhigh + ghostLayers;

    cout << "high(x,y,z): " << xhigh << "," << yhigh << "," << zhigh << endl;

    for (int k = l.z(); k < zhigh; k++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int i = l.x(); i < xhigh; i++) {
          cout << "(x,y,z): " << k << "," << j << "," << i << endl;
          // For an array of [ A ][ B ][ C ], we can index it thus:
          // (a * B * C) + (b * C) + (c * 1)
          int idx = i + (j * xstride) + (k * xstride * ystride);

          int xminus = (i - 1) + (j * xstride) + (k * xstride * ystride);
          int xplus  = (i + 1) + (j * xstride) + (k * xstride * ystride);
          int yminus = i + ((j - 1) * xstride) + (k * xstride * ystride);
          int yplus  = i + ((j + 1) * xstride) + (k * xstride * ystride);
          int zminus = i + (j * xstride) + ((k - 1) * xstride * ystride);
          int zplus  = i + (j * xstride) + ((k + 1) * xstride * ystride);

          newphi_data[idx] = (1. / 6) * (phi_data[xminus] + phi_data[xplus] + phi_data[yminus]
              + phi_data[yplus] + phi_data[zminus] + phi_data[zplus]);

          double diff = newphi_data[idx] - phi_data[idx];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
  } // end patch for loop
}

//______________________________________________________________________
//
void PoissonGPU1::timeAdvance3DP(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw) {

  int matl = 0;
  int ghostLayers = 1;

  // Do time steps
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector s = h - l;

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    //__________________________________
    //  3D-Pointer Stencil
    double*** phi_data = (double***)phi.getWindow()->getData()->get3DPointer();
    double*** newphi_data = (double***)newphi.getWindow()->getData()->get3DPointer();

    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();

    for (int i = l.z(); i < zhigh; i++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int k = l.x(); k < xhigh; k++) {

          double xminus = phi_data[i-1][j][k];
          double xplus  = phi_data[i+1][j][k];
          double yminus = phi_data[i][j-1][k];
          double yplus  = phi_data[i][j+1][k];
          double zminus = phi_data[i][j][k-1];
          double zplus  = phi_data[i][j][k+1];

          newphi_data[i][j][k] = (1. / 6) * (xminus + xplus + yminus + yplus + zminus + zplus);

          double diff = newphi_data[i][j][k] - phi_data[i][j][k];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
  } // end patch for loop
}

//______________________________________________________________________
//
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param domainHigh a three component vector that gives the highest non-ghost layer cell of the domain as (x,y,z)
// @param domainSize a three component vector that gives the size of the domain including ghost nodes
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
// @param residual the residual calculated by this individual kernel
__global__ void timeAdvanceKernel(uint3 domainLow,
                                  uint3 domainHigh,
                                  uint3 domainSize,
                                  int ghostLayers,
                                  double *phi,
                                  double *newphi,
                                  double *residual) {
  // calculate the thread indices
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  
  // Get the size of the data block in which the variables reside.
  //  This is essentially the stride in the index calculations.
  int dx = domainSize.x;
  int dy = domainSize.y;

  // If the threads are within the bounds of the ghost layers
  //  the algorithm is allowed to stream along the z direction
  //  applying the stencil to a line of cells.  The z direction
  //  is streamed because it allows access of x and y elements
  //  that are close to one another which should allow coalesced 
  //  memory accesses.
  if(i > 0 && j > 0 && i < domainHigh.x && j < domainHigh.y) {
    for (int k = domainLow.z; k < domainHigh.z; k++) {
      // For an array of [ A ][ B ][ C ], we can index it thus:
      // (a * B * C) + (b * C) + (c * 1)
      int idx = INDEX3D(dx,dy,i,j,k);

      newphi[idx] = (1. / 6) 
                  * (phi[INDEX3D(dx,dy, (i-1), j, k)] 
                   + phi[INDEX3D(dx,dy, (i+1), j, k)]
                   + phi[INDEX3D(dx,dy, i, (j-1), k)]
                   + phi[INDEX3D(dx,dy, i, (j+1), k)]
                   + phi[INDEX3D(dx,dy, i, j, (k-1))] 
                   + phi[INDEX3D(dx,dy, i, j, (k+1))]);

      // Still need a way to compute the residual as a reduction
      //  variable here.
    }
  }
}

//______________________________________________________________________
//
void PoissonGPU1::timeAdvanceGPU(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw) {

  int matl = 0;
  int previousPatchSize = 0;// this is to see if we need to release and reallocate between computations
  int size = 0;
  int ghostLayers = 1;

  // Device and host memor pointersy
  double* phi_host;
  double* phi_device;
  double* newphi_host;
  double* newphi_device;

  // Find the "best" device for cudaSetDevice() or in other words the device with 
  //  the highes number of processors.
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

  // Do time steps
  int numPatches = patches->size();
  for (int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    // Calculate the memory block size
    IntVector s = phi.getWindow()->getData()->size();
    int xdim = s.x(), ydim = s.y(), zdim = s.z();
    size = xdim * ydim * zdim * sizeof(double);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    // Check if we need to reallocate due to a change in the 
    //  size of this patch from the previous patch.
    if (size != previousPatchSize) {
      if (previousPatchSize != 0) {
        cudaFree(phi_device);
        cudaFree(newphi_device);
      }
      cudaMalloc(&phi_device, size);
      cudaMalloc(&newphi_device, size);
    }

    //___________________________________________
    //  Host->Device Memory Allocation and copy
    phi_host    = (double*)phi.getWindow()->getData()->getPointer();
    newphi_host = (double*)newphi.getWindow()->getData()->getPointer();

    CUDA_RT_SAFE_CALL(cudaMemcpy(phi_device,    phi_host,    size, cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALL(cudaMemcpy(newphi_device, newphi_host, size, cudaMemcpyHostToDevice));

    // Domain extents used by the kernel to prevent out of bounds accesses.
    uint3 domainLow  = make_uint3(l.x(), l.y(), l.z());
    uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());
    uint3 domainSize = make_uint3(s.x(), s.y(), s.z());

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


    // Launch kernel
    timeAdvanceKernel<<< totalBlocks, threadsPerBlock >>>(domainLow, domainHigh, domainSize, ghostLayers, phi_device, newphi_device, &residual);

    // Kernel error checking
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    } 



    //__________________________________
    //  Device->Host Memory Copy
    CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_RT_SAFE_CALL(cudaMemcpy(newphi_host, newphi_device, size, cudaMemcpyDeviceToHost));

    new_dw->put(sum_vartype(residual), residual_label);

  } // end patch for loop

  // free up allocated memory
  CUDA_RT_SAFE_CALL(cudaFree(phi_device));
  CUDA_RT_SAFE_CALL(cudaFree(newphi_device));

  
}
