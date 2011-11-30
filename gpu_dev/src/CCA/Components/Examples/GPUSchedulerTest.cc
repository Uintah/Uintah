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
  Task* gpuTask = scinew Task(&GPUSchedulerTest::initializeGPU, "GPUSchedulerTest::initializeGPU",
                           "GPUSchedulerTest::initialize", this, &GPUSchedulerTest::initialize);

  gpuTask->computes(phi_label);
  gpuTask->computes(residual_label);
  sched->addTask(gpuTask, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void GPUSchedulerTest::scheduleComputeStableTimestep(const LevelP& level, SchedulerP& sched) {
  Task* gpuTask = scinew Task(&GPUSchedulerTest::computeStableTimestepGPU, "GPUSchedulerTest::computeStableTimestepGPU",
                           "GPUSchedulerTest::computeStableTimestep", this, &GPUSchedulerTest::computeStableTimestep);

  gpuTask->requires(Task::NewDW, residual_label);
  gpuTask->computes(sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(gpuTask, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void GPUSchedulerTest::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched) {

//  // CPU task
//  Task* task = scinew Task("GPUSchedulerTest::timeAdvance", this, &GPUSchedulerTest::timeAdvance);
//  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
//  task->computes(phi_label);
//  task->computes(residual_label);
//  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  // GPU task
  Task* gpuTask = scinew Task(&GPUSchedulerTest::timeAdvanceGPU, "GPUSchedulerTest::timeAdvanceGPU",
                              "GPUSchedulerTest::timeAdvance", this, &GPUSchedulerTest::timeAdvance);
  gpuTask->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  gpuTask->computes(phi_label);
  gpuTask->computes(residual_label);
  sched->addTask(gpuTask, level->eachPatch(), sharedState_->allMaterials());
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
//  std::cout << std::endl << "\nIn computeStableTimestep\n" << std::endl;
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

//  std::cout << std::endl << "\nIn computeStableTimestepGPU\n" << std::endl;
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

#if 0
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor) {
      IntVector l,h;
      patch->getFaceNodes(Patch::xminus, 0, l, h);

      for(NodeIterator iter(l,h); !iter.done(); iter++) {
        if (phi[*iter] != 1.0) {
          cout << "phi_old[" << *iter << "]=" << phi[*iter] << endl;
        }
        phi[*iter]=1;
      }
    }
#endif

    new_dw->put(sum_vartype(-1), residual_label);
  }
//  std::cout << std::endl << "\nIn initialize\n" << std::endl;
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

#if 0
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor) {
      IntVector l,h;
      patch->getFaceNodes(Patch::xminus, 0, l, h);

      for(NodeIterator iter(l,h); !iter.done(); iter++) {
        if (phi[*iter] != 1.0) {
          cout << "phi_old[" << *iter << "]=" << phi[*iter] << endl;
        }
        phi[*iter]=1;
      }
    }
#endif

    new_dw->put(sum_vartype(-1), residual_label);
  }
//  std::cout << std::endl << "\nIn initializeGPU\n" << std::endl;
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

      //std::cout << "When cell is " << n << "   memory location is " << &(phi[n]) << std::endl;

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
//  std::cout << std::endl << "\nIn timeAdvance\n" << std::endl;
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

      //std::cout << "When cell is " << n << "   memory location is " << &(phi[n]) << std::endl;

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
//  std::cout << std::endl << "\nIn timeAdvanceGPU\n" << std::endl;
}
