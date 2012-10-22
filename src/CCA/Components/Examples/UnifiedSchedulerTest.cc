/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <CCA/Components/Examples/UnifiedSchedulerTest.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
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

UnifiedSchedulerTest::UnifiedSchedulerTest(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{

  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

UnifiedSchedulerTest::~UnifiedSchedulerTest()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::problemSetup(const ProblemSpecP& params,
                                        const ProblemSpecP& restart_prob_spec,
                                        GridP& grid,
                                        SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP gpuSchedTest = params->findBlock("UnifiedSchedulerTest");
  gpuSchedTest->require("delt", delt_);
  simpleMaterial_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(simpleMaterial_);
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleInitialize(const LevelP& level,
                                              SchedulerP& sched)
{
  Task* multiTask = scinew Task("UnifiedSchedulerTest::initialize", this, &UnifiedSchedulerTest::initialize);

  multiTask->computes(phi_label);
  multiTask->computes(residual_label);
  sched->addTask(multiTask, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleComputeStableTimestep(const LevelP& level,
                                                         SchedulerP& sched)
{
  Task* task = scinew Task("UnifiedSchedulerTest::computeStableTimestep", this, &UnifiedSchedulerTest::computeStableTimestep);

  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleTimeAdvance(const LevelP& level,
                                               SchedulerP& sched)
{
  Task* task = scinew Task(&UnifiedSchedulerTest::timeAdvanceGPU, "UnifiedSchedulerTest::timeAdvanceGPU",
                           "UnifiedSchedulerTest::timeAdvanceCPU", this, &UnifiedSchedulerTest::timeAdvanceCPU);
//  Task* task = scinew Task("GPUSchedulerTest::timeAdvanceCPU", this, &GPUSchedulerTest::timeAdvanceCPU);
//  Task* task = scinew Task("GPUSchedulerTest::timeAdvance1DP", this, &GPUSchedulerTest::timeAdvance1DP);
//  Task* task = scinew Task("GPUSchedulerTest::timeAdvance3DP", this, &GPUSchedulerTest::timeAdvance3DP);

  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::computeStableTimestep(const ProcessorGroup* pg,
                                                 const PatchSubset* patches,
                                                 const MaterialSubset* matls,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw)
{
  if (pg->myrank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(), getLevel(patches));
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::initialize(const ProcessorGroup* pg,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face)) {
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
void UnifiedSchedulerTest::timeAdvanceCPU(const ProcessorGroup* pg,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
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
                  * (phi[n + IntVector(1, 0, 0)] + phi[n + IntVector(-1, 0, 0)] + phi[n + IntVector(0, 1, 0)]
                     + phi[n + IntVector(0, -1, 0)] + phi[n + IntVector(0, 0, 1)] + phi[n + IntVector(0, 0, -1)]);

      double diff = newphi[n] - phi[n];
      residual += diff * diff;
    }
    new_dw->put(sum_vartype(residual), residual_label);
  }
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvance1DP(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{

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

//    cout << "high(x,y,z): " << xhigh << "," << yhigh << "," << zhigh << endl;

    for (int k = l.z(); k < zhigh; k++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int i = l.x(); i < xhigh; i++) {
          cout << "(x,y,z): " << k << "," << j << "," << i << endl;
          // For an array of [ A ][ B ][ C ], we can index it thus:
          // (a * B * C) + (b * C) + (c * 1)
          int idx = i + (j * xstride) + (k * xstride * ystride);

          int xminus = (i - 1) + (j * xstride) + (k * xstride * ystride);
          int xplus = (i + 1) + (j * xstride) + (k * xstride * ystride);
          int yminus = i + ((j - 1) * xstride) + (k * xstride * ystride);
          int yplus = i + ((j + 1) * xstride) + (k * xstride * ystride);
          int zminus = i + (j * xstride) + ((k - 1) * xstride * ystride);
          int zplus = i + (j * xstride) + ((k + 1) * xstride * ystride);

          newphi_data[idx] = (1. / 6)
                             * (phi_data[xminus] + phi_data[xplus] + phi_data[yminus] + phi_data[yplus] + phi_data[zminus]
                                + phi_data[zplus]);

          double diff = newphi_data[idx] - phi_data[idx];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
  }  // end patch for loop
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvance3DP(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{

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

//    cout << "high(x,y,z): " << xhigh << "," << yhigh << "," << zhigh << endl;

    for (int i = l.z(); i < zhigh; i++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int k = l.x(); k < xhigh; k++) {

          double xminus = phi_data[i - 1][j][k];
          double xplus = phi_data[i + 1][j][k];
          double yminus = phi_data[i][j - 1][k];
          double yplus = phi_data[i][j + 1][k];
          double zminus = phi_data[i][j][k - 1];
          double zplus = phi_data[i][j][k + 1];

          newphi_data[i][j][k] = (1. / 6) * (xminus + xplus + yminus + yplus + zminus + zplus);

          double diff = newphi_data[i][j][k] - phi_data[i][j][k];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
  }  // end patch for loop
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvanceGPU(const ProcessorGroup* pg,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw,
                                          int device)
{

  // setup for driver API kernel launch
  CUresult cuErrVal;
  CUmodule cuModule;
  CUfunction gpuSchedulerTestKernel;

  // initialize the driver API
  CUDA_DRV_SAFE_CALL( cuErrVal = cuInit(0))

  // set the CUDA device and context
  CUDA_RT_SAFE_CALL( cudaSetDevice(device));

  // get a handle on the GPU scheduler to query for device and host pointers, etc
  UnifiedScheduler* sched = dynamic_cast<UnifiedScheduler*>(getPort("scheduler"));

  // Do time steps
  int NGC = 1;
  int matl = 0;

  // requisite pointers
  double* d_phi = NULL;
  double* d_newphi = NULL;

  int numPatches = patches->size();
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
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    // Domain extents used by the kernel to prevent out of bounds accesses.
    uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
    uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());
    uint3 domainSize = make_uint3(s.x(), s.y(), s.z());

    // Set up number of thread blocks in X and Y directions accounting for dimensions not divisible by 8
    int xBlocks = ((xdim % 8) == 0) ? (xdim / 8) : ((xdim / 8) + 1);
    int yBlocks = ((ydim % 8) == 0) ? (ydim / 8) : ((ydim / 8) + 1);
    dim3 gridDim(xBlocks, yBlocks, 1);  // grid dimensions (blocks per grid))

    int tpbX = 8;
    int tpbY = 8;
    int tpbZ = 1;
    dim3 dimBlock(tpbX, tpbY, tpbZ);  // block dimensions (threads per block)

    // setup and launch kernel
    void *kernelParms[] = { &domainLow, &domainHigh, &domainSize, &NGC, &d_phi, &d_newphi };
    string ptxpath = string(PTX_DIR_PATH) + "/GPUSchedulerTestKernel.ptx";
    CUDA_DRV_SAFE_CALL( cuErrVal = cuModuleLoad(&cuModule, ptxpath.c_str()));
    CUDA_DRV_SAFE_CALL( cuErrVal = cuModuleGetFunction(&gpuSchedulerTestKernel, cuModule, "gpuSchedulerTestKernel"));
    cudaStream_t* stream = sched->getCudaStream(device);

    // launch the kernel
    cuErrVal = cuLaunchKernel(gpuSchedulerTestKernel, gridDim.x, gridDim.y, gridDim.z, dimBlock.x, dimBlock.y, dimBlock.z, 0,
                              *stream, kernelParms, 0);

    // get the results back to the host
    cudaEvent_t* event = sched->getCudaEvent(device);
    sched->requestD2HCopy(phi_label, matl, patch, stream, event);

    new_dw->put(sum_vartype(residual), residual_label);

  }  // end patch for loop
}
