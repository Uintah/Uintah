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

#include <CCA/Components/Examples/GPUSchedulerTest.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
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

GPUSchedulerTest::GPUSchedulerTest(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld) {

  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

GPUSchedulerTest::~GPUSchedulerTest() {
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void GPUSchedulerTest::problemSetup(const ProblemSpecP& params,
                                    const ProblemSpecP& restart_prob_spec,
                                    GridP& grid,
                                    SimulationStateP& sharedState) {
  sharedState_ = sharedState;
  ProblemSpecP gpuSchedTest = params->findBlock("GPUSchedulerTest");
  gpuSchedTest->require("delt", delt_);
  simpleMaterial_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(simpleMaterial_);
}
//______________________________________________________________________
//
void GPUSchedulerTest::scheduleInitialize(const LevelP& level, SchedulerP& sched) {
  Task* multiTask = scinew Task(&GPUSchedulerTest::initializeGPU, "GPUSchedulerTest::initializeGPU",
                                "GPUSchedulerTest::initialize", this, &GPUSchedulerTest::initialize);

  multiTask->computes(phi_label);
  multiTask->computes(residual_label);
  sched->addTask(multiTask, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void GPUSchedulerTest::scheduleComputeStableTimestep(const LevelP& level, SchedulerP& sched) {
  Task* task = scinew Task("GPUSchedulerTest::computeStableTimestep", this, &GPUSchedulerTest::computeStableTimestep);

  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void GPUSchedulerTest::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched) {
  Task* multiTask = scinew Task(&GPUSchedulerTest::timeAdvanceGPU, "GPUSchedulerTest::timeAdvanceGPU",
                                "GPUSchedulerTest::timeAdvance", this, &GPUSchedulerTest::timeAdvance);

  multiTask->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  multiTask->computes(phi_label);
  multiTask->computes(residual_label);
  sched->addTask(multiTask, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
//
void GPUSchedulerTest::computeStableTimestep(const ProcessorGroup* pg,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse*  old_dw,
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
void GPUSchedulerTest::computeStableTimestepGPU(const ProcessorGroup* pg,
                                                const PatchSubset* patches,
                                                const MaterialSubset* matls,
                                                DataWarehouse*  old_dw,
                                                DataWarehouse* new_dw,
                                                int device) {
  if (pg->myrank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}

//______________________________________________________________________
//
void GPUSchedulerTest::initialize(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw) {
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face = Patch::nextFace(face)) {
      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr,
                                                             child);
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
void GPUSchedulerTest::initializeGPU(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     int device) {
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face = Patch::nextFace(face)) {
      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr,
                                                             child);
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
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainSize a three component vector that gives the size of the domain as (x,y,z)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
__global__ void timeAdvanceTestKernel(uint3 domainSize,
                                      uint3 domainLower,
                                      int NGC,
                                      double *phi,
                                      double *newphi) {

// calculate the thread indices
  int tidX = blockDim.x * blockIdx.x + threadIdx.x;
  int tidY = blockDim.y * blockIdx.y + threadIdx.y;

  int num_slices = domainSize.z - NGC;
  int dx = domainSize.x;
  int dy = domainSize.y;

  if (tidX < (dx - NGC) && tidY < (dy - NGC) && tidX > 0 && tidY > 0) {
    for (int slice = NGC; slice < num_slices; slice++) {

      //__________________________________
      //  GPU Stencil
      newphi[INDEX3D(dx, dy, tidX, tidY, slice)] =
          (1.0 / 6.0)
          * (phi[INDEX3D(dx, dy, tidX - 1, tidY, slice)]
             + phi[INDEX3D(dx, dy, tidX + 1, tidY, slice)]
             + phi[INDEX3D(dx, dy, tidX, tidY - 1, slice)]
             + phi[INDEX3D(dx, dy, tidX, tidY + 1, slice)]
             + phi[INDEX3D(dx, dy, tidX, tidY, slice - 1)]
             + phi[INDEX3D(dx, dy, tidX, tidY, slice + 1)]);
    }
  }
}

//______________________________________________________________________
//
void GPUSchedulerTest::timeAdvance(const ProcessorGroup* pg,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw) {
  int matl = 0;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;

    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
    NCVariable<double> newphi;

    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    //__________________________________
    //  Stencil
    for (NodeIterator iter(l, h); !iter.done(); iter++) {
      IntVector n = *iter;

      newphi[n] = (1. / 6)
                  * (phi[n + IntVector(1, 0, 0)] + phi[n + IntVector(-1, 0, 0)]
                     + phi[n + IntVector(0, 1, 0)]
                     + phi[n + IntVector(0, -1, 0)]
                     + phi[n + IntVector(0, 0, 1)]
                     + phi[n + IntVector(0, 0, -1)]);

      double diff = newphi[n] - phi[n];
      residual += diff * diff;
    }
    new_dw->put(sum_vartype(residual), residual_label);
  }
}

//______________________________________________________________________
//
void GPUSchedulerTest::timeAdvanceGPU(const ProcessorGroup* pg,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      int device) {
  int matl = 0;
  int ghostLayers = 1;

  // declare device and host memory
  double* newphi_device;
  double* phi_device;
  double* phi_host;
  double* phi_hostPinned;
  double* newphi_host;
  double* newphi_hostPinned;

  CUDA_SAFE_CALL( cudaSetDevice(device) );

  // Do time steps
  int previousPatchSize = 0; // see if we need to release and reallocate between computations
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
    int xdim = s.x(), ydim = s.y(), zdim = s.z();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
        patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    // check if we need to reallocate
    int nbytes = xdim * ydim * zdim * sizeof(double);
    if (nbytes != previousPatchSize) {
      if (previousPatchSize != 0) {
        CUDA_SAFE_CALL( cudaFree(phi_device) );
        CUDA_SAFE_CALL( cudaFree(newphi_device) );
      }
      // use host alloc so we can use pinned mem for async API
      cutilSafeCall( cudaMallocHost((void**)&phi_hostPinned, nbytes) );
      cutilSafeCall( cudaMallocHost((void**)&newphi_hostPinned, nbytes) );

      // now the malloc device mem
      CUDA_SAFE_CALL( cudaMalloc((void**)&phi_device, nbytes) );
      CUDA_SAFE_CALL( cudaMalloc((void**)&newphi_device, nbytes) );
    }

    //  Memory Allocation
    phi_host = (double*)phi.getWindow()->getData()->getPointer();
    CUDA_SAFE_CALL( cudaMemcpy(phi_hostPinned, phi_host, nbytes, cudaMemcpyHostToHost) );
    newphi_host = (double*)newphi.getWindow()->getData()->getPointer();
    CUDA_SAFE_CALL( cudaMemcpy(newphi_hostPinned, newphi_host, nbytes, cudaMemcpyHostToHost) );

    // configure kernel launch
    uint3 domainSize = make_uint3(xdim, ydim, zdim);
    uint3 domainLower = make_uint3(l.x(), l.y(), l.z());
    int totalBlocks = nbytes / (sizeof(double) * xdim * ydim * zdim);
    dim3 threadsPerBlock(xdim, ydim, zdim);
    if (nbytes % (totalBlocks) != 0) {
      totalBlocks++;
    }

    // async copy host2device -> launch kernel -> async host2device
    cudaMemcpyAsync(phi_device, phi_hostPinned, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(newphi_device, newphi_hostPinned, nbytes, cudaMemcpyHostToDevice);
    timeAdvanceTestKernel<<< totalBlocks, threadsPerBlock >>>(domainSize, domainLower, ghostLayers, phi_device, newphi_device);
    cudaMemcpyAsync(newphi_host, newphi_device, nbytes, cudaMemcpyDeviceToHost);

    new_dw->put(sum_vartype(residual), residual_label);

  }  // end patch for loop

  // free up allocated memory
  cudaFree(phi_device);
  cudaFreeHost(phi_hostPinned);
  cudaFree(newphi_device);
  cudaFreeHost(newphi_hostPinned);
}
