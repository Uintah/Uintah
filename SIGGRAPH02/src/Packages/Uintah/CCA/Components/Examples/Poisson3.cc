
#include <Packages/Uintah/CCA/Components/Examples/Poisson3.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

/********************/
/* The main program */
/********************/

Poisson3::Poisson3(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld),
  interpolator_(2)
{
  lb_ = scinew ExamplesLabel();
}

Poisson3::~Poisson3()
{
  delete lb_;
}

void Poisson3::problemSetup(const ProblemSpecP& params, GridP&,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");
  poisson->require("delt", delt_);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void Poisson3::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  Task* task = scinew Task("initialize",
			   this, &Poisson3::initialize);
  task->computes(lb_->phi);
  task->computes(lb_->residual);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void Poisson3::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Poisson3::computeStableTimestep);
  task->requires(Task::NewDW, lb_->residual);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &Poisson3::timeAdvance);
  task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, 1);
  task->computes(lb_->phi);
  task->computes(lb_->residual);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::computeStableTimestep(const ProcessorGroup* pg,
				  const PatchSubset*,
				  const MaterialSubset*,
				  DataWarehouse*, DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, lb_->residual);
    cerr << "Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label());
}

void Poisson3::initialize(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse*, DataWarehouse* new_dw)
{
  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      new_dw->allocateAndPut(phi, lb_->phi, matl, patch);
      phi.initialize(0);
      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
	IntVector l,h;
	patch->getFace(Patch::xminus, 0, l, h);
	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  phi[*iter]=1;
      }
      new_dw->put(sum_vartype(-1), lb_->residual);
    }
  }
}

void Poisson3::timeAdvance(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      constNCVariable<double> phi;
      old_dw->get(phi, lb_->phi, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;
      new_dw->allocateAndPut(newphi, lb_->phi, matl, patch);
      newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
      double residual = 0;
      IntVector l = patch->getNodeLowIndex();
      IntVector h = patch->getNodeHighIndex(); 
      l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
      h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
		     patch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);
      for(NodeIterator iter(l, h); !iter.done(); iter++){
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
    }
  }
}




void Poisson3::scheduleRefine(LevelP& coarseLevel, LevelP& fineLevel, SchedulerP& sched)
{
  PatchSubset* coarsePatches = (PatchSubset*) (coarseLevel->allPatches()->getUnion());
  PatchSubset* finePatches = (PatchSubset*) (fineLevel->allPatches()->getUnion());

  Task* task = scinew Task("refine", this, &Poisson3::refine, coarseLevel);
  task->requires(Task::OldDW, lb_->phi, coarsePatches, Task::OutOfDomain, 0, Task::NormalDomain, 
                 Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  task->computes(lb_->phi);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}


void Poisson3::refine(const ProcessorGroup*,
                      const PatchSubset* finePatches, 
		      const MaterialSubset* matls,
                      DataWarehouse* coarseDW,
                      DataWarehouse* fineDW,
                      LevelP coarseLevel)   // should be const, but there's no getPatchFromPoint for const Levels)
{
  // For all patches
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);
    Point low = finePatch->nodePosition(finePatch->getNodeLowIndex());
    const Patch* coarsePatch = coarseLevel->getPatchFromPoint(low);

    // For all materials
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      constNCVariable<double> coarsePhi;
      coarseDW->get(coarsePhi, lb_->phi, matl, coarsePatch, Ghost::AroundNodes, interpolator_.getMaxSupportRefine());

      NCVariable<double> finePhi;
      fineDW->allocateAndPut(finePhi, lb_->phi, matl, finePatch);

      IntVector l = finePatch->getNodeLowIndex();
      IntVector h = finePatch->getNodeHighIndex(); 
      // For all finegrid nodes
      for(NodeIterator iter(l, h); !iter.done(); iter++){
//	finePhi[*iter] = interpolator_.refine(coarsePhi, *iter, Interpolator::inner);
      }
    }
  }

  // Set the new delt
  fineDW->put(delt_vartype(delt_), sharedState_->get_delt_label());
}


void Poisson3::scheduleRefineInterface(LevelP& coarseLevel,
			               LevelP& fineLevel,
			               SchedulerP& sched)
{
  PatchSubset* coarsePatches = (PatchSubset*) (coarseLevel->allPatches()->getUnion());
  PatchSubset* finePatches = (PatchSubset*) (fineLevel->allPatches()->getUnion());

  Task* task = scinew Task("refineInterface", this, &Poisson3::refine, coarseLevel);
  task->requires(Task::OldDW, lb_->phi, coarsePatches, Task::OutOfDomain, 0, Task::NormalDomain, 
                 Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  task->computes(lb_->phi);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}


void Poisson3::refineInterface(const ProcessorGroup*,
	                       const PatchSubset* finePatches, 
		               const MaterialSubset* matls,
                               DataWarehouse* fineDW, 
		               DataWarehouse* coarseDW,
                               LevelP coarseLevel)
{
  // For all patches
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);
    Point low = finePatch->nodePosition(finePatch->getNodeLowIndex());
    const Patch* coarsePatch = coarseLevel->getPatchFromPoint(low);

    if(finePatch->getBCType(Patch::xminus) != Patch::Neighbor
       || finePatch->getBCType(Patch::yminus) != Patch::Neighbor
       || finePatch->getBCType(Patch::zminus) != Patch::Neighbor
       || finePatch->getBCType(Patch::xplus)  != Patch::Neighbor
       || finePatch->getBCType(Patch::yplus)  != Patch::Neighbor
       || finePatch->getBCType(Patch::zplus)  != Patch::Neighbor)
    {
      // For all materials
      for(int m = 0; m < matls->size(); m++){
	int matl = matls->get(m);
	constNCVariable<double> coarsePhi;
	coarseDW->get(coarsePhi, lb_->phi, matl, coarsePatch, Ghost::AroundNodes, interpolator_.getMaxSupportRefine());

	NCVariable<double> finePhi;
        fineDW->getModifiable(finePhi, lb_->phi, matl, finePatch);

	// For all Interfaces:
        IntVector l,h;
        for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face))
	{
	  if(finePatch->getBCType(face) != Patch::Neighbor)
	  {
	    finePatch->getFace(face, 0, l, h);

	    // For all finegrid nodes
	    for(NodeIterator iter(l, h); !iter.done(); iter++){
//	      finePhi[*iter] = interpolator_.refine(coarsePhi, *iter, Interpolator::inner);
	    }
	  }
        }
      }
    }
  }

  // Set the new delt
  fineDW->put(delt_vartype(delt_), sharedState_->get_delt_label());
}


void Poisson3::scheduleCoarsen(LevelP& coarseLevel, LevelP& fineLevel, SchedulerP& sched)
{
  PatchSubset* coarsePatches = (PatchSubset*) coarseLevel->allPatches()->getUnion();
  PatchSubset* finePatches = (PatchSubset*) fineLevel->allPatches()->getUnion();

  Task* task = scinew Task("coarsen", this, &Poisson3::coarsen, coarseLevel);
  task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, interpolator_.getMaxSupportCoarsen());
  task->modifies(lb_->phi, coarsePatches, Task::OutOfDomain, 0, Task::NormalDomain);
  //task->requires(Task::NewDW, lb_->phi, coarsePatches, Task::OutOfDomain, 0, Task::NormalDomain, Ghost::None, 0);
  //task->computes(lb_->phi, coarsePatches, Task::OutOfDomain, 0, Task::NormalDomain);
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}




void Poisson3::coarsen(const ProcessorGroup*,
	               const PatchSubset* finePatches, 
		       const MaterialSubset* matls,
                       DataWarehouse* fineDW, 
		       DataWarehouse* coarseDW,
                       LevelP coarseLevel)   // should be const, but there's no getPatchFromPoint for const Levels
{
  // For all patches
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);
    Point low = finePatch->nodePosition(finePatch->getNodeLowIndex());
    const Patch* coarsePatch = coarseLevel->getPatchFromPoint(low);

    // For all materials
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      constNCVariable<double> finePhi;
      fineDW->get(finePhi, lb_->phi, matl, finePatch, Ghost::AroundNodes, interpolator_.getMaxSupportCoarsen());

      NCVariable<double> coarsePhi;
      coarseDW->getModifiable(coarsePhi, lb_->phi, matl, coarsePatch);

      IntVector fl = finePatch->getNodeLowIndex();
      IntVector l;
      coarsePatch->findClosestNode(finePatch->nodePosition(fl), l);

      //IntVector fh = finePatch->getNodeHighIndex(); 
      IntVector h;
      coarsePatch->findClosestNode(finePatch->nodePosition(fl), h);

      // For all coarsegrid nodes
      for(NodeIterator iter(l, h); !iter.done(); iter++){
//	coarsePhi[*iter] = interpolator_.coarsen(finePhi, *iter, Interpolator::inner);
      }

      coarseDW->put(coarsePhi, lb_->phi, matl, coarsePatch);
    }
  }
}


