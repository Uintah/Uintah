
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
#include <Core/Util/DebugStream.h>
#include <Core/Malloc/Allocator.h>
#include <iomanip>

using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("Poisson3", false);

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
  if(level->getIndex() == 0){
    dbg << "scheduleInitialize\n";
    Task* task = scinew Task("initialize",
			     this, &Poisson3::initialize);
    task->computes(lb_->phi);
    task->computes(lb_->residual, level.get_rep());
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  } else {
    scheduleRefine(level, sched);
  }
}
 
void Poisson3::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &Poisson3::computeStableTimestep);
  task->requires(Task::NewDW, lb_->residual, level.get_rep());
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}


void
Poisson3::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched,
			       int, int )
{
  Task* task = scinew Task("timeAdvance",
			   this, &Poisson3::timeAdvance,
			   level->getIndex() != 0);
  if(level->getIndex() == 0) {
    task->requires(Task::OldDW, lb_->phi, Ghost::AroundNodes, 1);
    task->computes(lb_->phi);
  } else {
    task->requires(Task::NewDW, lb_->phi, Ghost::AroundNodes, 1);
    task->modifies(lb_->phi);
  }
  task->computes(lb_->residual, level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::computeStableTimestep(const ProcessorGroup* pg,
				     const PatchSubset* pss,
				     const MaterialSubset*,
				     DataWarehouse*, DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, lb_->residual, getLevel(pss));
    cout << "Level " << getLevel(pss)->getIndex() << ": Residual=" << residual << '\n';
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
	patch->getFaceNodes(Patch::xminus, 0, l, h);
	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  phi[*iter]=1;
      }
      new_dw->put(sum_vartype(-1), lb_->residual, patch->getLevel());
    }
  }
}

void Poisson3::timeAdvance(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw, DataWarehouse* new_dw,
			   bool modify)
{
  dbg << "Poisson3::timeAdvance\n";
  DataWarehouse* fromDW = modify?new_dw:old_dw;
  for(int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    dbg << "timeAdvance on patch: " << *patch << '\n';
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      NCVariable<double> phi;
      fromDW->getCopy(phi, lb_->phi, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;
      if(modify) {
	new_dw->getModifiable(newphi, lb_->phi, matl, patch);
      } else {
	new_dw->allocateAndPut(newphi, lb_->phi, matl, patch);
	newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());
      }
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
      new_dw->put(sum_vartype(residual), lb_->residual, patch->getLevel());
    }
  }
}

void Poisson3::scheduleRefine(const LevelP& fineLevel, SchedulerP& sched)
{
  dbg << "Poisson3::scheduleRefine\n";
  Task* task = scinew Task("refine", this, &Poisson3::refine);
  task->requires(Task::NewDW, lb_->phi,
		 0, Task::CoarseLevel,
		 0, Task::NormalDomain, 
                 Ghost::AroundCells, interpolator_.getMaxSupportRefine());
  task->computes(lb_->phi);
  task->computes(lb_->residual, fineLevel.get_rep());
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::refine(const ProcessorGroup*,
                      const PatchSubset* finePatches, 
		      const MaterialSubset* matls,
                      DataWarehouse*,
                      DataWarehouse* newDW)
{
  dbg << "Poisson3::refine\n";
  if(finePatches->size() == 0)
    return;
  const Level* fineLevel = finePatches->get(0)->getLevel();
  LevelP coarseLevel = fineLevel->getCoarserLevel();
  // For all patches
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);
    IntVector low = finePatch->getNodeLowIndex();
    IntVector high = finePatch->getNodeHighIndex();
    // Find the overlapping regions...
    Patch::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      int total_fine = 0;
      NCVariable<double> finePhi;
      newDW->allocateAndPut(finePhi, lb_->phi, matl, finePatch);
      // For each coarse patch, compute the overlapped region and interpolate
      for(int i=0;i<coarsePatches.size();i++){
	const Patch* coarsePatch = coarsePatches[i];
	constNCVariable<double> coarsePhi;
	newDW->get(coarsePhi, lb_->phi, matl, coarsePatch,
		   Ghost::AroundCells, interpolator_.getMaxSupportRefine());

	IntVector l = Max(coarseLevel->mapNodeToFiner(coarsePatch->getNodeLowIndex()),
			  finePatch->getNodeLowIndex());
	IntVector h = Min(coarseLevel->mapNodeToFiner(coarsePatch->getNodeHighIndex()),
			  finePatch->getNodeHighIndex());
	IntVector diff = h-l;
	total_fine += diff.x()*diff.y()*diff.z();
	// For all finegrid nodes
	// This is pretty inefficient.  It should be changed to loop over
	// coarse grid nodes instead and then have a small loop inside?
	// - Steve
	for(NodeIterator iter(l, h); !iter.done(); iter++){
	  finePhi[*iter] = interpolator_.refine(coarsePhi, *iter, Interpolator::Inner);
	}
      }
      IntVector diff = high-low;
      ASSERTEQ(total_fine, diff.x()*diff.y()*diff.z());
    }
    newDW->put(sum_vartype(-1), lb_->residual, finePatch->getLevel());
  }

  dbg << "Poisson3::refine done\n";
}


void Poisson3::scheduleRefineInterface(const LevelP& fineLevel,
				       SchedulerP& sched,
				       int step, int nsteps)
{
  dbg << "Poisson3::scheduleRefineInterface\n";
  Task* task = scinew Task("refineInterface", this, &Poisson3::refineInterface,
			   step, nsteps);

  task->requires(Task::OldDW, lb_->phi, Ghost::None);
  task->requires(Task::CoarseOldDW, lb_->phi,
		 0, Task::CoarseLevel,
		 0, Task::NormalDomain, 
                 Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  if(step != 0)
    task->requires(Task::CoarseNewDW, lb_->phi,
		   0, Task::CoarseLevel,
		   0, Task::NormalDomain, 
		   Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  task->computes(lb_->phi);
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}


void Poisson3::refineInterface(const ProcessorGroup*,
	                       const PatchSubset* finePatches, 
		               const MaterialSubset* matls,
                               DataWarehouse* old_dw, 
		               DataWarehouse* new_dw,
			       int step, int nsteps)
{
  dbg << "Poisson3::refineInterface\n";
  // Doesn't interpolate between coarse DWs
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  dbg << "old_dw: " << old_dw->getID() << ", new_dw: " << new_dw->getID() << ", coarse_old_dw: " << coarse_old_dw->getID() << ", coarse_new_dw: " << coarse_new_dw->getID() << '\n';
  const Level* fineLevel = getLevel(finePatches);
  LevelP coarseLevel = fineLevel->getCoarserLevel();
  double weight1 = double(step)/double(nsteps);
  double weight2 = 1-weight1;
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      constNCVariable<double> phi;
      old_dw->get(phi, lb_->phi, matl, finePatch, Ghost::None, 0);
      NCVariable<double> finePhi;
      new_dw->allocateAndPut(finePhi, lb_->phi, matl, finePatch);
      finePhi.copyPatch(phi, finePhi.getLowIndex(), finePhi.getHighIndex());

      for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
	   face = Patch::nextFace(face)) {
	if(finePatch->getBCType(face) != Patch::Neighbor) {
	  IntVector low,high;
	  finePatch->getFaceNodes(face, 0, low, high);
	  IntVector coarseLow = fineLevel->mapNodeToCoarser(low);
	  IntVector coarseHigh = fineLevel->mapNodeToCoarser(high);

	  // Find the overlapping regions...
	  Patch::selectType coarsePatches;
	  coarseLevel->selectPatches(coarseLow, coarseHigh, coarsePatches);

	  int total_fine = 0;
	  // For each coarse patch, compute the overlapped region and interpolate
	  for(int i=0;i<coarsePatches.size();i++){
	    const Patch* coarsePatch = coarsePatches[i];
	    IntVector l = Max(coarseLevel->mapNodeToFiner(coarsePatch->getNodeLowIndex()),
			      low);
	    IntVector h = Min(coarseLevel->mapNodeToFiner(coarsePatch->getNodeHighIndex()),
			      high);
	    IntVector diff = h-l;
	    total_fine += diff.x()*diff.y()*diff.z();
	    if(step == 0){
	      // For all finegrid nodes
	      constNCVariable<double> coarsePhi;
	      coarse_old_dw->get(coarsePhi, lb_->phi, matl, coarsePatch,
				 Ghost::AroundCells,
				 interpolator_.getMaxSupportRefine());
	      for(NodeIterator iter(l, h); !iter.done(); iter++){
		finePhi[*iter] = interpolator_.refine(coarsePhi, *iter, Interpolator::Inner);
	      }
	    } else {
	      constNCVariable<double> coarsePhi1;
	      coarse_old_dw->get(coarsePhi1, lb_->phi, matl, coarsePatch,
				 Ghost::AroundCells,
				 interpolator_.getMaxSupportRefine());
	      constNCVariable<double> coarsePhi2;
	      coarse_new_dw->get(coarsePhi2, lb_->phi, matl, coarsePatch,
				 Ghost::AroundCells,
				 interpolator_.getMaxSupportRefine());
	      for(NodeIterator iter(l, h); !iter.done(); iter++){
		finePhi[*iter] = interpolator_.refine(coarsePhi1, weight1, coarsePhi2, weight2, *iter, Interpolator::Inner);
	      }
	    }
	  }
	  IntVector diff = high-low;
	  ASSERTEQ(total_fine, diff.x()*diff.y()*diff.z());
	}
      }
    }
  }
}


void Poisson3::scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched)
{
  Task* task = scinew Task("coarsen", this, &Poisson3::coarsen);
  task->requires(Task::NewDW, lb_->phi,
		 0, Task::FineLevel,
		 0, Task::NormalDomain,
		 Ghost::AroundNodes, interpolator_.getMaxSupportCoarsen());
  task->modifies(lb_->phi);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::coarsen(const ProcessorGroup*,
	               const PatchSubset* coarsePatches,
		       const MaterialSubset* matls,
                       DataWarehouse*, 
		       DataWarehouse* newDW)
{
  if(coarsePatches->size() == 0)
    return;
  const Level* coarseLevel = coarsePatches->get(0)->getLevel();
  LevelP fineLevel = coarseLevel->getFinerLevel();

  // For all patches
  for(int p = 0; p < coarsePatches->size(); p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    IntVector low = coarsePatch->getNodeLowIndex();
    IntVector high = coarsePatch->getNodeHighIndex();
    IntVector fine_low = coarseLevel->mapNodeToFiner(low);
    IntVector fine_high = coarseLevel->mapNodeToFiner(high);

    // Find the overlapping regions...
    Patch::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    // For all materials
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      NCVariable<double> coarsePhi;
      newDW->getModifiable(coarsePhi, lb_->phi, matl, coarsePatch);

      // For each fine patch, compute the overlapped region and interpolate
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	constNCVariable<double> finePhi;
	newDW->get(finePhi, lb_->phi, matl, finePatch,
		   Ghost::AroundNodes, interpolator_.getMaxSupportCoarsen());
	IntVector l = Max(fineLevel->mapNodeToCoarser(finePatch->getNodeLowIndex()),
			  coarsePatch->getNodeLowIndex());
	IntVector h = Min(fineLevel->mapNodeToCoarser(finePatch->getNodeHighIndex()),
			  coarsePatch->getNodeHighIndex());
	l += IntVector(finePatch->getBCType(Patch::xminus) == Patch::Neighbor?0:1,
		       finePatch->getBCType(Patch::yminus) == Patch::Neighbor?0:1,
		       finePatch->getBCType(Patch::zminus) == Patch::Neighbor?0:1);
	h -= IntVector(finePatch->getBCType(Patch::xplus) == Patch::Neighbor?0:1,
		       finePatch->getBCType(Patch::yplus) == Patch::Neighbor?0:1,
		       finePatch->getBCType(Patch::zplus) == Patch::Neighbor?0:1);
	// For all coarsegrid nodes
	for(NodeIterator iter(l, h); !iter.done(); iter++){
	  coarsePhi[*iter] = interpolator_.coarsen(finePhi, *iter, Interpolator::Inner);
	}
      }
    }
  }
}
