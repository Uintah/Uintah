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
#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>
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
  Task* multiTask = scinew Task("GPUSchedulerTest::initialize", this, &GPUSchedulerTest::initialize);

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
// @brief A kernel that applies the stencil used in timeAdvance(...)
// @param domainLower a three component vector that gives the lower corner of the work area as (x,y,z)
// @param domainHigh a three component vector that gives the highest non-ghost layer cell of the domain as (x,y,z)
// @param domainSize a three component vector that gives the size of the domain including ghost nodes
// @param ghostLayers the number of layers of ghost cells
// @param phi pointer to the source phi allocated on the device
// @param newphi pointer to the sink phi allocated on the device
// @param residual the residual calculated by this individual kernel
__global__ void timeAdvanceTestKernel(uint3 domainLow,
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

      // Still need a way to compute the residual as a reduction variable here.
    }
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

  // set the CUDA context
  cudaError_t retVal;
  CUDA_SAFE_CALL( retVal = cudaSetDevice(device) );

  // get a handle on the GPU scheduler to query for device and host pointers, etc
  GPUThreadedMPIScheduler* sched = dynamic_cast<GPUThreadedMPIScheduler*>(getPort("scheduler"));

  // Do time steps
  int NGC = 1;
  int numPatches = patches->size();
  int matl = 0;

  // requisite pointers
  double* d_phi = NULL;
  double* d_newphi = NULL;

  for (int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);
    double residual = 0;

    d_phi = sched->getDeviceRequiresPtr(phi_label, matl, patch);
    d_newphi = sched->getDeviceComputesPtr(phi_label, matl, patch);

    // Calculate the memory block size
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();
    IntVector s = sched->getDeviceRequiresSize(phi_label, matl, patch);
    int xdim = s.x(), ydim = s.y();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor ? 0 : 1);


    // Domain extents used by the kernel to prevent out of bounds accesses.
    uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
    uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());
    uint3 domainSize = make_uint3(s.x(), s.y(), s.z());

    // Threads per block must be power of 2 in each direction.  Here
    //  8 is chosen as a test value in the x and y and 1 in the z,
    //  as each of these (x,y) threads streams through the z direction.
    dim3 threadsPerBlock(8, 8, 1);

    // Set up the number of blocks of threads in each direction accounting for any
    //  non-power of 8 end pieces.
    int xBlocks = xdim / 8;
    if (xdim % 8 != 0) {
      xBlocks++;
    }
    int yBlocks = ydim / 8;
    if (ydim % 8 != 0) {
      yBlocks++;
    }
    dim3 totalBlocks(xBlocks, yBlocks);

    // setup and launch kernel
    cudaStream_t* stream = sched->getCudaStream(phi_label, matl, patch);
    cudaEvent_t* event = sched->getCudaEvent(phi_label, matl, patch);
    timeAdvanceTestKernel<<< totalBlocks, threadsPerBlock, 0, *stream >>>(domainLow,
                                                                          domainHigh,
                                                                          domainSize,
                                                                          NGC,
                                                                          d_phi,
                                                                          d_newphi,
                                                                          &residual);

    // Kernel error checking (for now)
    retVal = cudaGetLastError();
    if (retVal != cudaSuccess) {
      fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
      exit(-1);
    }

    // get the results back to the host
    sched->requestD2HCopy(phi_label, matl, patch, stream, event);

    new_dw->put(sum_vartype(residual), residual_label);

  }  // end patch for loop
}
