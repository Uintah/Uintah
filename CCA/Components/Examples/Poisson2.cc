
#include <Packages/Uintah/CCA/Components/Examples/Poisson2.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

Poisson2::Poisson2(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

Poisson2::~Poisson2()
{
  delete lb_;
}

void Poisson2::problemSetup(const ProblemSpecP& params, GridP& grid,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");
  poisson->require("delt", delt_);
  poisson->require("maxresidual", maxresidual_);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void Poisson2::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Poisson2::initialize);
  task->computes(lb_->phi);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void Poisson2::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Poisson2::computeStableTimestep);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Poisson2::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &Poisson2::timeAdvance,
			   level, sched);
  task->hasSubScheduler();
  task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, 1);
  task->computes(lb_->phi);
  LoadBalancer* lb = sched->getLoadBalancer();
  const PatchSet* perproc_patches = lb->createPerProcessorPatchSet(level, d_myworld);
  sched->addTask(task, perproc_patches, sharedState_->allMaterials());
}

void Poisson2::computeStableTimestep(const ProcessorGroup* pg,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label());
}

void Poisson2::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      new_dw->allocate(phi, lb_->phi, matl, patch);
      phi.initialize(0);
      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
	IntVector l,h;
	patch->getFace(Patch::xminus, 0, l, h);
	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  phi[*iter]=1;
      }
      new_dw->put(phi, lb_->phi, matl, patch);
    }
  }
}

void Poisson2::timeAdvance(const ProcessorGroup* pg,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw, DataWarehouse* new_dw,
			   LevelP level, SchedulerP sched)
{
  cerr << "Poisson2 time advance!\n";
  SchedulerP subsched = sched->createSubScheduler();
  subsched->initialize();
  GridP grid = level->getGrid();
  subsched->advanceDataWarehouse(grid);
  // Put the initial data
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      old_dw->get(phi, lb_->phi, matl, patch, Ghost::None, 0);
      subsched->get_new_dw()->put(phi, lb_->phi, matl, patch);
    }
  }

  subsched->advanceDataWarehouse(grid);
  // Create the tasks
  Task* task = scinew Task("iterate",
			   this, &Poisson2::iterate);
  task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, 1);
  task->computes(lb_->phi);
  subsched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  // Compile the scheduler
  subsched->compile(d_myworld, false);

  // Iterate
  int count = 0;
  double residual;
  do {
    subsched->execute(d_myworld);
    sum_vartype residual_var;
    subsched->get_new_dw()->get(residual_var, lb_->residual);
    residual = residual_var;

    if(pg->myrank() == 0)
      cerr << "Iteration " << count++ << ", residual=" << residual << '\n';
    subsched->advanceDataWarehouse(grid);
  } while(residual > maxresidual_);

  // Get the final data
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      // DW has been advanced, so it is old now
      subsched->get_old_dw()->get(phi, lb_->phi, matl, patch, Ghost::None, 0);
      new_dw->put(phi, lb_->phi, matl, patch);
    }
  }
  cerr << "Poisson2 time advance done!\n";
}


void Poisson2::iterate(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  cerr << "Poisson iterate!\n";
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      old_dw->get(phi, lb_->phi, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;
      new_dw->allocate(newphi, lb_->phi, matl, patch);
      double residual=0;
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex(); 
      newphi.copy(phi, l, h);
      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);
      for(NodeIterator iter(l, h);!iter.done(); iter++){
	newphi[*iter]=(1./6)*(
	  phi[*iter+IntVector(1,0,0)]+phi[*iter+IntVector(-1,0,0)]+
	  phi[*iter+IntVector(0,1,0)]+phi[*iter+IntVector(0,-1,0)]+
	  phi[*iter+IntVector(0,0,1)]+phi[*iter+IntVector(0,0,-1)]);
	double diff = newphi[*iter]-phi[*iter];
	residual += diff*diff;
      }
#if 0
      foreach boundary that exists on this patch {
	count face boundaries;
	switch(kind of bc for boundary){
	case Dirichlet:
	  set the value accordingly;
	  break;
	case Neumann:
	  Do the different derivative;
	  break;
	}
      }
      ASSERT(numFaceBoundaries == patch->getNumBoundaryFaces());
#endif
      new_dw->put(sum_vartype(residual), lb_->residual);
      new_dw->put(newphi, lb_->phi, matl, patch);
    }
  }
}
