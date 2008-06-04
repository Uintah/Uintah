
#include <Packages/Uintah/CCA/Components/Examples/Poisson1.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Poisson1::Poisson1(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

Poisson1::~Poisson1()
{
  delete lb_;
}
//______________________________________________________________________
//
void Poisson1::problemSetup(const ProblemSpecP& params, 
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
void Poisson1::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::initialize",
                     this, &Poisson1::initialize);
                     
  task->computes(lb_->phi);
  task->computes(lb_->residual);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Poisson1::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::computeStableTimestep",
                     this, &Poisson1::computeStableTimestep);
                     
  task->requires(Task::NewDW, lb_->residual);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void
Poisson1::scheduleTimeAdvance( const LevelP& level, 
                               SchedulerP& sched)
{
  Task* task = scinew Task("Poisson1::timeAdvance",
                     this, &Poisson1::timeAdvance);
                     
  task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, 1);
  task->computes(lb_->phi);
  task->computes(lb_->residual);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void Poisson1::computeStableTimestep(const ProcessorGroup* pg,
                                     const PatchSubset* /*patches*/,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, lb_->residual);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label());
}

//______________________________________________________________________
//
void Poisson1::initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, lb_->phi, matl, patch);
    phi.initialize(0);
 
    if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
       IntVector l,h;
       patch->getFaceNodes(Patch::xminus, 0, l, h);
 
      for(NodeIterator iter(l,h); !iter.done(); iter++){
         phi[*iter]=1;
      }
    }
    new_dw->put(sum_vartype(-1), lb_->residual);
  }
}
//______________________________________________________________________
//
void Poisson1::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw)
{
  int matl = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
 
    old_dw->get(phi, lb_->phi, matl, patch, Ghost::AroundNodes, 1);
    NCVariable<double> newphi;
 
    new_dw->allocateAndPut(newphi, lb_->phi, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
 
    double residual=0;
    IntVector l = patch->getNodeLowIndex__New();
    IntVector h = patch->getNodeHighIndex__New();
 
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
    new_dw->put(sum_vartype(residual), lb_->residual);
  }
}
