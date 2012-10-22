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


#include <CCA/Components/Examples/Poisson3.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <iomanip>

using namespace std;
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
  phi_label = VarLabel::create("phi", 
                               NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", 
                                    sum_vartype::getTypeDescription());
}

Poisson3::~Poisson3()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}

void Poisson3::problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP&, SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP poisson = params->findBlock("Poisson");
  poisson->require("delt", delt_);
  mymat_ = scinew SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);
}
 
void Poisson3::scheduleInitialize(const LevelP& level,
				  SchedulerP& sched)
{
  if(level->getIndex() == 0){
    dbg << "scheduleInitialize\n";
    Task* task = scinew Task("initialize",
			     this, &Poisson3::initialize);
    task->computes(phi_label);
    task->computes(residual_label, level.get_rep());
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
  task->requires(Task::NewDW, residual_label, level.get_rep());
  task->computes(sharedState_->get_delt_label(),level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}


void
Poisson3::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  Task* task = scinew Task("timeAdvance",
			   this, &Poisson3::timeAdvance,
			   level->getIndex() != 0);
  if(level->getIndex() == 0) {
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->computes(phi_label);
  } else {
    task->requires(Task::NewDW, phi_label, Ghost::AroundNodes, 1);
    task->modifies(phi_label);
  }
  task->computes(residual_label, level.get_rep());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void Poisson3::computeStableTimestep(const ProcessorGroup* pg,
				     const PatchSubset* pss,
				     const MaterialSubset*,
				     DataWarehouse*, DataWarehouse* new_dw)
{
  if(pg->myrank() == 0){
    sum_vartype residual;
    new_dw->get(residual, residual_label, getLevel(pss));
    cout << "Level " << getLevel(pss)->getIndex() << ": Residual=" << residual << '\n';
  }
  new_dw->put(delt_vartype(delt_), sharedState_->get_delt_label(),getLevel(pss));
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
      new_dw->allocateAndPut(phi, phi_label, matl, patch);
      phi.initialize(0);
      for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
           face=Patch::nextFace(face)) {
        
        if (patch->getBCType(face) == Patch::None) {
          int numChildren = 
            patch->getBCDataArray(face)->getNumberChildren(matl);
          for (int child = 0; child < numChildren; child++) {
            Iterator nbound_ptr, nu;
            
            const BoundCondBase* bcb = patch->getArrayBCValues(face,matl,"Phi",
                                                               nu,nbound_ptr,
                                                               child);
            
            const BoundCond<double>* bc = 
              dynamic_cast<const BoundCond<double>*>(bcb); 
            double value = bc->getValue();
            for (nbound_ptr.reset(); !nbound_ptr.done();nbound_ptr++) {
              phi[*nbound_ptr]=value;
              
            }
          }
        }
      }    
#if 0
      if(patch->getBCType(Patch::xminus) != Patch::Neighbor){
	IntVector l,h;
	patch->getFaceNodes(Patch::xminus, 0, l, h);
	for(NodeIterator iter(l,h); !iter.done(); iter++)
	  phi[*iter]=1;
      }
#endif
      new_dw->put(sum_vartype(-1), residual_label, patch->getLevel());
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
      fromDW->getCopy(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
      NCVariable<double> newphi;
      if(modify) {
	new_dw->getModifiable(newphi, phi_label, matl, patch);
      } else {
	new_dw->allocateAndPut(newphi, phi_label, matl, patch);
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
      new_dw->put(sum_vartype(residual), residual_label, patch->getLevel());
    }
  }
}

void Poisson3::scheduleRefine(const LevelP& fineLevel, SchedulerP& sched)
{
  dbg << "Poisson3::scheduleRefine\n";
  Task* task = scinew Task("refine", this, &Poisson3::refine);
  task->requires(Task::NewDW, phi_label,
		 0, Task::CoarseLevel,
		 0, Task::NormalDomain, 
                 Ghost::AroundCells, interpolator_.getMaxSupportRefine());
  task->computes(phi_label);
  task->computes(residual_label, fineLevel.get_rep());
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
      newDW->allocateAndPut(finePhi, phi_label, matl, finePatch);
      // For each coarse patch, compute the overlapped region and interpolate
      for(int i=0;i<coarsePatches.size();i++){
	const Patch* coarsePatch = coarsePatches[i];
	constNCVariable<double> coarsePhi;
	newDW->get(coarsePhi, phi_label, matl, coarsePatch,
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
    newDW->put(sum_vartype(-1), residual_label, finePatch->getLevel());
  }

  dbg << "Poisson3::refine done\n";
}


void Poisson3::scheduleRefineInterface(const LevelP& fineLevel,
				       SchedulerP& sched, bool needCoarseOld, bool needCoarseNew)
{
  dbg << "Poisson3::scheduleRefineInterface\n";
  Task* task = scinew Task("refineInterface", this, &Poisson3::refineInterface);

  task->requires(Task::OldDW, phi_label, Ghost::None);
  task->requires(Task::CoarseOldDW, phi_label,
		 0, Task::CoarseLevel,
		 0, Task::NormalDomain, 
                 Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  if(needCoarseNew)
    task->requires(Task::CoarseNewDW, phi_label,
		   0, Task::CoarseLevel,
		   0, Task::NormalDomain, 
		   Ghost::AroundNodes, interpolator_.getMaxSupportRefine());
  task->computes(phi_label);
  sched->addTask(task, fineLevel->eachPatch(), sharedState_->allMaterials());
}


void Poisson3::refineInterface(const ProcessorGroup*,
	                       const PatchSubset* finePatches, 
		               const MaterialSubset* matls,
                               DataWarehouse* old_dw, 
		               DataWarehouse* new_dw)
{
  dbg << "Poisson3::refineInterface\n";
  // Doesn't interpolate between coarse DWs
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  dbg << "old_dw: " << old_dw->getID() << ", new_dw: " << new_dw->getID() << ", coarse_old_dw: " << coarse_old_dw->getID() << ", coarse_new_dw: " << coarse_new_dw->getID() << '\n';
  const Level* fineLevel = getLevel(finePatches);
  LevelP coarseLevel = fineLevel->getCoarserLevel();
  double weight1 = getSubCycleProgress(new_dw);
  double weight2 = 1-weight1;
  for(int p = 0; p < finePatches->size(); p++){
    const Patch* finePatch = finePatches->get(p);

    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      constNCVariable<double> phi;
      old_dw->get(phi, phi_label, matl, finePatch, Ghost::None, 0);
      NCVariable<double> finePhi;
      new_dw->allocateAndPut(finePhi, phi_label, matl, finePatch);
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
	    if(weight1 == 0){
	      // For all finegrid nodes
	      constNCVariable<double> coarsePhi;
	      coarse_old_dw->get(coarsePhi, phi_label, matl, coarsePatch,
				 Ghost::AroundCells,
				 interpolator_.getMaxSupportRefine());
	      for(NodeIterator iter(l, h); !iter.done(); iter++){
		finePhi[*iter] = interpolator_.refine(coarsePhi, *iter, Interpolator::Inner);
	      }
	    } else {
	      constNCVariable<double> coarsePhi1;
	      coarse_old_dw->get(coarsePhi1, phi_label, matl, coarsePatch,
				 Ghost::AroundCells,
				 interpolator_.getMaxSupportRefine());
	      constNCVariable<double> coarsePhi2;
	      coarse_new_dw->get(coarsePhi2, phi_label, matl, coarsePatch,
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
  task->requires(Task::NewDW, phi_label,
		 0, Task::FineLevel,
		 0, Task::NormalDomain,
		 Ghost::AroundNodes, interpolator_.getMaxSupportCoarsen());
  task->modifies(phi_label);
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
    // Not Used: IntVector fine_low = coarseLevel->mapNodeToFiner(low);
    // Not Used: IntVector fine_high = coarseLevel->mapNodeToFiner(high);

    // Find the overlapping regions...
    Patch::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    // For all materials
    for(int m = 0; m < matls->size(); m++){
      int matl = matls->get(m);
      NCVariable<double> coarsePhi;
      newDW->getModifiable(coarsePhi, phi_label, matl, coarsePatch);

      // For each fine patch, compute the overlapped region and interpolate
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	constNCVariable<double> finePhi;
	newDW->get(finePhi, phi_label, matl, finePatch,
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
