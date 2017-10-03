/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifdef HAVE_CUDA
//#include <CCA/Components/Schedulers/GPUUtilities.h>
#include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#endif

#include <sci_defs/cuda_defs.h>

#define BLOCKSIZE 16

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
  ProblemSpecP unifiedSchedTest = params->findBlock("UnifiedSchedulerTest");
  unifiedSchedTest->require("delt", delt_);
  simpleMaterial_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(simpleMaterial_);
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleInitialize(const LevelP& level,
                                              SchedulerP& sched)
{
  Task* multiTask = scinew Task("UnifiedSchedulerTest::initialize", this, &UnifiedSchedulerTest::initialize);

  multiTask->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  //multiTask->computes(phi_label);
  multiTask->computes(residual_label);
  sched->addTask(multiTask, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleRestartInitialize(const LevelP& level,
                                                     SchedulerP& sched)
{
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
  Task* task = scinew Task("UnifiedSchedulerTest::timeAdvanceUnified", this, &UnifiedSchedulerTest::timeAdvanceUnified);
  //Task* task = scinew Task("UnifiedSchedulerTest::timeAdvanceCPU", this, &UnifiedSchedulerTest::timeAdvanceCPU);
  //Task* task = scinew Task("UnifiedSchedulerTest::timeAdvance1DP", this, &UnifiedSchedulerTest::timeAdvance1DP);
  //Task* task = scinew Task("UnifiedSchedulerTest::timeAdvance3DP", this, &UnifiedSchedulerTest::timeAdvance3DP);

#ifdef HAVE_CUDA
  if (Uintah::Parallel::usingDevice()) {
    task->usesDevice(true);
  }
#endif

  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
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
                                      DataWarehouse* /*old_dw*/,
                                      DataWarehouse* new_dw)
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    //new_dw->allocateAndPut(phi, phi_label, matl, patch);
    new_dw->allocateAndPut(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
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
    //printf("cpu - phi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 9, patch->getID(), phi[IntVector(0, 1, 9)]);
    //printf("cpu - phi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 10, patch->getID(), phi[IntVector(0, 1, 10)]);
    new_dw->put(sum_vartype(-1), residual_label);
  }
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvanceUnified(DetailedTask* task,
                                              Task::CallBackEvent event,
                                              const ProcessorGroup* pg,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw,
                                              void* old_TaskGpuDW,
                                              void* new_TaskGpuDW,
                                              void* stream,
                                              int deviceID)
{
  // When Task is scheduled to CPU
  if (event == Task::CPU) {
    int matl = 0;

    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      constNCVariable<double> phi;

      old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
      //printf("after cpu - phi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 9, patch->getID(), phi[IntVector(0, 1, 9)]);
      //printf("after cpu - phi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 10, patch->getID(), phi[IntVector(0, 1, 10)]);
      NCVariable<double> newphi;

      new_dw->allocateAndPut(newphi, phi_label, matl, patch);
      newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
      //printf("cpu - newphi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 9, patch->getID(), newphi[IntVector(0, 1, 9)]);
      //printf("cpu - newphi(%d, %d, %d) for patch %d is %1.6lf\n", 0, 1, 10, patch->getID(), newphi[IntVector(0, 1, 10)]);
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
        //if (n.x() == 1 && n.y() == 1 && n.z() == 1) {
        //  printf("cpu - newphi(%d, %d, %d) is %1.6lf from %1.6lf %1.6lf %1.6lf %1.6lf %1.6lf %1.6lf\n", n.x(), n.y(), n.z(), newphi[n], phi[n + IntVector(-1, 0, 0)], phi[n + IntVector(1, 0, 0)], phi[n + IntVector(0, -1, 0)], phi[n + IntVector(0, 1, 0)], phi[n + IntVector(0, 0, -1)], phi[n + IntVector(0, 0, 1)]);
        //}
        //if (n.x() == 1 && n.y() == 1 && (n.z() == 9 || n.z() == 10)) {
        //  printf("cpu - newphi(%d, %d, %d) is %1.6lf from %1.6lf %1.6lf %1.6lf %1.6lf %1.6lf %1.6lf\n", n.x(), n.y(), n.z(), newphi[n], phi[n + IntVector(-1, 0, 0)], phi[n + IntVector(1, 0, 0)], phi[n + IntVector(0, -1, 0)], phi[n + IntVector(0, 1, 0)], phi[n + IntVector(0, 0, -1)], phi[n + IntVector(0, 0, 1)]);
        //}
        double diff = newphi[n] - phi[n];
        residual += diff * diff;
      }
      new_dw->put(sum_vartype(residual), residual_label);
    }
  }  //end CPU

  // When Task is scheduled to GPU
  if (event == Task::GPU) {
    // Do time steps
    int numPatches = patches->size();
    for (int p = 0; p < numPatches; p++) {
      const Patch* patch = patches->get(p);

      // Calculate the memory block size
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex();

      uint3 patchNodeLowIndex = make_uint3(l.x(), l.y(), l.z());
      uint3 patchNodeHighIndex = make_uint3(h.x(), h.y(), h.z());
      IntVector s = h - l;
      int xdim = s.x();
      int ydim = s.y();

      // define dimensions of the thread grid to be launched
      int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
      int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
      dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
      dim3 dimGrid(xblocks, yblocks, 1);
      //dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
      //dim3 dimGrid(xblocks, yblocks, 1);

      //now calculate the computation domain (ignoring the outside cell regions)

      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
              patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
              patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
              patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
              patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

      // Domain extents used by the kernel to prevent out of bounds accesses.
      uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
      uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());

      // setup and launch kernel
      //GPUDataWarehouse* old_gpudw = old_dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch))->getdevice_ptr();
      //GPUDataWarehouse* new_gpudw = new_dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch))->getdevice_ptr();
      GPUGridVariable<double> device_var;
      new_dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch))->get(device_var, "phi", patch->getID(), 0, 0);
        //void* device_ptr = nullptr;
        //device_var.getArray3(device_offset, device_size, device_ptr);
        //device_ptr = device_var.getPointer();
        //printf("Calling unifiedSchedulerTestKernel for (%d,%d,%d) to (%d,%d,%d) with device variable at %p on stream %p on threadID %d\n", patchNodeLowIndex.x,patchNodeLowIndex.y,patchNodeLowIndex.z, patchNodeHighIndex.x, patchNodeHighIndex.y, patchNodeHighIndex.z, device_ptr, stream, Uintah::Thread::self()->myid());

      launchUnifiedSchedulerTestKernel(dimGrid,
                                       dimBlock,
                                       (cudaStream_t*) stream,
                                       patch->getID(),
                                       patchNodeLowIndex,
                                       patchNodeHighIndex,
                                       domainLow,
                                       domainHigh,
                                       (GPUDataWarehouse*)old_TaskGpuDW,
                                       (GPUDataWarehouse*)new_TaskGpuDW);

      // residual is automatically "put" with the D2H copy of the GPUReductionVariable
      // new_dw->put(sum_vartype(residual), residual_label);

    } // end patch for loop
  } //end GPU
    /*
    int numPatches = patches->size();
        for (int p = 0; p < numPatches; p++) {

          const Patch* patch = patches->get(p);

          // Calculate the memory block size
          IntVector l = patch->getNodeLowIndex();
          IntVector h = patch->getNodeHighIndex();

          uint3 patchNodeLowIndex = make_uint3(l.x(), l.y(), l.z());
          uint3 patchNodeHighIndex = make_uint3(h.x(), h.y(), h.z());
          IntVector s = h - l;
          int xdim = s.x();
          int ydim = s.y();

          // define dimensions of the thread grid to be launched
          int xblocks = (int)ceil((float)xdim / BLOCKSIZE);
          int yblocks = (int)ceil((float)ydim / BLOCKSIZE);
          dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
          dim3 dimGrid(xblocks, yblocks, 1);
          //dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
          //dim3 dimGrid(xblocks, yblocks, 1);

          //now calculate the computation domain (ignoring the outside cell regions)

          l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
          h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

          // Domain extents used by the kernel to prevent out of bounds accesses.
          uint3 domainLow = make_uint3(l.x(), l.y(), l.z());
          uint3 domainHigh = make_uint3(h.x(), h.y(), h.z());

          // setup and launch kernel
          GPUDataWarehouse* old_gpudw = old_dw->getGPUDW()->getdevice_ptr();
          GPUDataWarehouse* new_gpudw = new_dw->getGPUDW()->getdevice_ptr();
          GPUGridVariable<double> device_var;
          new_dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch))->get(device_var, "phi", patch->getID(), 0);
            int3 device_offset;
            int3 device_size;
            //void* device_ptr = nullptr;
            //device_var.getArray3(device_offset, device_size, device_ptr);
            //device_ptr = device_var.getPointer();

          //printf("Finished unifiedSchedulerTestKernel for (%d,%d,%d) to (%d,%d,%d) with device variable at %p on stream %p on threadID %d\n", patchNodeLowIndex.x,patchNodeLowIndex.y,patchNodeLowIndex.z, patchNodeHighIndex.x, patchNodeHighIndex.y, patchNodeHighIndex.z, device_ptr, stream, Uintah::Thread::self()->myid());
    }*/
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
          //cout << "(x,y,z): " << k << "," << j << "," << i << endl;
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
    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();

    for (int i = l.z(); i < zhigh; i++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int k = l.x(); k < xhigh; k++) {
          double xminus = phi(i-1,j,k);
          double xplus  = phi(i+1,j,k);
          double yminus = phi(i,j-1,k);
          double yplus  = phi(i,j+1,k);
          double zminus = phi(i,j,k-1);
          double zplus  = phi(i,j,k+1);

          newphi(i,j,k) = (1. / 6) * (xminus + xplus + yminus + yplus + zminus + zplus);

          double diff = newphi(i,j,k) - phi(i,j,k);
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), residual_label);
  }  // end patch for loop
}
