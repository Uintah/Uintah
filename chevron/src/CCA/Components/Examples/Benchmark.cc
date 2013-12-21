/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <CCA/Components/Examples/Benchmark.h>
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

Benchmark::Benchmark(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{

  phi_label = VarLabel::create("phi", 
                               NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", 
                                    sum_vartype::getTypeDescription());
}

Benchmark::~Benchmark()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}
//______________________________________________________________________
//
void Benchmark::problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& /*grid*/, 
                            SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");
  
  poisson->require("delt", delt_);
  
  mymat_ = scinew SimpleMaterial();
  
  sharedState->registerSimpleMaterial(mymat_);
}
//______________________________________________________________________
//
void Benchmark::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  Task* task = scinew Task("Benchmark::initialize",
                     this, &Benchmark::initialize);
                     
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Benchmark::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  Task* task = scinew Task("Benchmark::computeStableTimestep",
                     this, &Benchmark::computeStableTimestep);
                     
  task->requires(Task::NewDW, residual_label);
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
Benchmark::scheduleTimeAdvance( const LevelP& level, 
                               SchedulerP& sched)
{
  Task* task = scinew Task("Benchmark::timeAdvance",
                     this, &Benchmark::timeAdvance);
                     
  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Benchmark::computeStableTimestep(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, residual_label);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(patches));
}

//______________________________________________________________________
//
void Benchmark::initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
         face=Patch::nextFace(face)) {

      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;
          
          const BoundCondBase* bcb = patch->getArrayBCValues(face,matl,"Phi",nu,
                                                             nbound_ptr,child);
          
          const BoundCond<double>* bc = 
            dynamic_cast<const BoundCond<double>*>(bcb); 
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++) {
            phi[*nbound_ptr]=value;

          }
          delete bcb;
        }
      }
    }            
#if 0
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
       IntVector l,h;
       patch->getFaceNodes(Patch::xminus, 0, l, h);
 
      for(NodeIterator iter(l,h); !iter.done(); iter++){
        if (phi[*iter] != 1.0) {
          cout << "phi_old[" << *iter << "]=" << phi[*iter] << endl;
        }
         phi[*iter]=1;
      }
    }
#endif

    new_dw->put(sum_vartype(-1), residual_label);
  }
}
//______________________________________________________________________
//
void Benchmark::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
 
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
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
  }
}
