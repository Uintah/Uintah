#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/Schedulers/LocallyComputedPatchVarMap.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/MaterialSetP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

using namespace Uintah;

PatchCombiner::PatchCombiner(const ProcessorGroup* world, string udaDir)
  : UintahParallelComponent(world), udaDir_(udaDir), dataArchive_(0),
    dw_(0), world_(world), timeIndex_(0)
{
  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());
}

PatchCombiner::~PatchCombiner()
{
  delete dw_;
  delete dataArchive_;
  
  VarLabel::destroy(delt_label);
  for (unsigned int i = 0; i < labels_.size(); i++)
    VarLabel::destroy(labels_[i]);
}

void PatchCombiner::problemSetup(const ProblemSpecP& /*params*/, GridP& grid,
				 SimulationStateP& state)
{
  d_sharedState = state;

  
  if (world_->size() > 1) {
    throw InternalError("combine_patches not yet supported in parallel");
  }  
  dataArchive_ = scinew DataArchive(udaDir_, world_->myrank(), world_->size());
  dataArchive_->queryTimesteps(timesteps_, times_);
  DataArchive::cacheOnlyCurrentTimestep = true;

  // try to add a time to times_ that will get us passed the end of the
  // simulation -- yes this is a hack!!
  times_.push_back(10 * times_[times_.size() - 1]);

  GridP newGrid = scinew Grid();
  GridP oldGrid = dataArchive_->queryGrid(times_[0]);

  const SuperPatchContainer* superPatches;
  for (int i = 0; i < oldGrid->numLevels(); i++) {
    LevelP level = oldGrid->getLevel(i);
    LocallyComputedPatchVarMap patchGrouper;
    const PatchSubset* patches = level->allPatches()->getUnion();
    patchGrouper.addComputedPatchSet(0, patches);
    patchGrouper.makeGroups();
    superPatches = patchGrouper.getSuperPatches(0, level.get_rep());
    ASSERT(superPatches != 0);

    LevelP newLevel = newGrid->addLevel(level->getAnchor(), level->dCell());

    SuperPatchContainer::const_iterator superIter;
    for (superIter = superPatches->begin(); superIter != superPatches->end();
	 superIter++) {
      IntVector low = (*superIter)->getLow();
      IntVector high = (*superIter)->getHigh();
      IntVector inLow = high; // taking min values starting at high
      IntVector inHigh = low; // taking max values starting at low
      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
	const Patch* patch = (*superIter)->getBoxes()[p];
	inLow = Min(inLow, patch->getInteriorCellLowIndex());
	inHigh = Max(inHigh, patch->getInteriorCellHighIndex());
      }
      
      Patch* newPatch =
	newLevel->addPatch(low, high, inLow, inHigh);
      for (unsigned int p = 0; p < (*superIter)->getBoxes().size(); p++) {
	const Patch* patch = (*superIter)->getBoxes()[p];
	new2OldPatchMap_[newPatch].push_back(patch);
      }
    }
    
    newLevel->finalizeLevel();
  }
    
  oldGrid_ = oldGrid;
  grid = newGrid;
}

void PatchCombiner::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  // labels_ should be empty, but just in case...
  for (unsigned int i = 0; i < labels_.size(); i++)
    VarLabel::destroy(labels_[i]);
  labels_.clear();

  vector<string> names;
  vector< const TypeDescription *> typeDescriptions;
  dataArchive_->queryVariables(names, typeDescriptions);
  for (unsigned int i = 0; i < names.size(); i++) {
    labels_.push_back(VarLabel::create(names[i], typeDescriptions[i]));
  }

  Task* t = scinew Task("PatchCombiner::initialize", this, &PatchCombiner::initialize);
  t->computes(delt_label);
  MaterialSetP globalMatlSet = scinew MaterialSet();
  globalMatlSet->add(-1);
  sched->addTask(t, level->eachPatch() /* ??? -- what to do for multi-proc */,
		 globalMatlSet.get_rep());
}

void
PatchCombiner::scheduleTimeAdvance( const LevelP& level,
				    SchedulerP& sched, int, int )
{
  double time = times_[timeIndex_];
  const PatchSet* eachPatch = level->eachPatch();
    
  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;
  for (unsigned int i = 0; i < labels_.size(); i++) {
    VarLabel* label = labels_[i];
    Task* t;
    if (label->typeDescription()->getType() !=
	TypeDescription::ParticleVariable)
      t = scinew Task("PatchCombiner::setGridVars", this, &PatchCombiner::setGridVars, label);
    else
      t = scinew Task("PatchCombiner::setParticleVars", this, &PatchCombiner::setParticleVars, label);

    ConsecutiveRangeSet matlsRangeSet;
    for (int i = 0; i < eachPatch->size(); i++) {
      const Patch* patch = eachPatch->getSubset(i)->get(0);
      list<const Patch*>& oldPatches = new2OldPatchMap_[patch];
      for (list<const Patch*>::iterator iter = oldPatches.begin();
	   iter != oldPatches.end(); iter++) {
	const Patch* oldPatch = *iter;
	matlsRangeSet = matlsRangeSet.
	  unioned(dataArchive_->queryMaterials(label->getName(), 
					       oldPatch, time));
      }
    }
    MaterialSet* matls;
    if (prevMatlSet != 0 && prevRangeSet == matlsRangeSet) {
      matls = prevMatlSet.get_rep();
    }
    else {
      matls = scinew MaterialSet();
      vector<int> matls_vec;
      matls_vec.reserve(matlsRangeSet.size());
      for (ConsecutiveRangeSet::iterator iter = matlsRangeSet.begin();
	   iter != matlsRangeSet.end(); iter++) {
	matls_vec.push_back(*iter);
      }
      matls->addAll(matls_vec);
      prevRangeSet = matlsRangeSet;
      prevMatlSet = matls;
    }

    t->computes(label, matls->getUnion());
    // require delt just to set up the dependency (what it really needs is
    // dw_).
    t->requires(Task::NewDW, delt_label);    
    sched->addTask(t, eachPatch, matls);
  }

  Task* readTask = scinew Task("PatchCombiner::readAndSetDelT", this, &PatchCombiner::readAndSetDelT, sched.get_rep());
  readTask->computes(delt_label);
  MaterialSetP globalMatlSet = scinew MaterialSet();
  globalMatlSet->add(-1);
  sched->addTask(readTask, eachPatch /* ??? -- what to do for multi-proc */,
		 globalMatlSet.get_rep());
} // end scheduleTimeAdvance()

void PatchCombiner::initialize(const ProcessorGroup*,
			       const PatchSubset* /*patches*/,
			       const MaterialSubset* /*matls*/,
			       DataWarehouse* /*old_dw*/,
			       DataWarehouse* new_dw)
{
  double t = times_[0];
  delt_vartype delt_var = t; /* should subtract off start time --
				this assumes it's 0 */
  new_dw->put(delt_var, delt_label);  
}


void PatchCombiner::readAndSetDelT(const ProcessorGroup*,
				   const PatchSubset* /*patches*/,
				   const MaterialSubset* /*matls*/,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw,
				   Scheduler* /*sched*/)
{
  int generation = new_dw->getID();

  // The timestep index should be the same as the new_dw generation number - 1
  // (this is a bit of a hack, but it should work).
  timeIndex_ = generation - 1;  
  double time = times_[timeIndex_];
  //cerr << "The time is now: " << time << endl;
  int timestep = timesteps_[timeIndex_];
  
  delete dw_;
  dw_=scinew OnDemandDataWarehouse(0, 0, generation, oldGrid_);
  //cerr << "deleted and recreated dw\n";
  double delt;
  dataArchive_->restartInitialize(timestep, oldGrid_, dw_, &time, &delt);
  //cerr << "restartInitialize done\n";
  
  // don't use that delt -- jump to the next output timestep
  delt = times_[timeIndex_ + 1] - time;
  delt_vartype delt_var = delt;
  new_dw->put(delt_var, delt_label);
  //cerr << "done with reading\n";
}

void PatchCombiner::setGridVars(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* /*old_dw*/,
				DataWarehouse* new_dw,
				VarLabel* label)
{
  // just get delt even though it's just a dummy requires to set up the
  // dependency for dw_ -- get it so it doesn't complain about requiring
  // something not used.
  delt_vartype delt_var;  
  new_dw->get(delt_var, delt_label);
	      
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      Variable* var = label->typeDescription()->createInstance();
      new_dw->allocateAndPutGridVar(var, label, matl, patch);

      // fill in the new patch with each of the old patches that made it up.
      list<const Patch*>& oldPatches = new2OldPatchMap_[patch];
      for (list<const Patch*>::iterator iter = oldPatches.begin();
	   iter != oldPatches.end(); iter++) {
	dw_->copyOutGridData(var, label, matl, *iter); 
      }

      delete var; // a clone should have been put in the dw
      //new_dw->put(var, label, matl, patch);	
    }
  }
}

void PatchCombiner::setParticleVars(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw,
				    VarLabel* label)
{
  // just get delt even though it's just a dummy requires to set up the
  // dependency for dw_ -- get it so it doesn't complain about requiring
  // something not used.
  delt_vartype delt_var;  
  new_dw->get(delt_var, delt_label);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      Variable* var = label->typeDescription()->createInstance();
      ParticleVariableBase* pvar = dynamic_cast<ParticleVariableBase*>(var);
      
      // fill in the new patch with each of the old patches that made it up.
      list<const Patch*>& oldPatches = new2OldPatchMap_[patch];

      // get the particle count
      particleIndex numParticles = 0;
      for (list<const Patch*>::iterator iter = oldPatches.begin();
	   iter != oldPatches.end(); iter++) {
	//cerr << "Getting particle subset from patch " << *iter << endl;
	numParticles += dw_->getParticleSubset(matl, *iter)->numParticles();
      }

      ParticleSubset* psubset = 0;      
      if (new_dw->haveParticleSubset(matl, patch)) {
	psubset = new_dw->getParticleSubset(matl, patch);
	ASSERT(psubset->numParticles() == numParticles);
      }
      else {
	psubset = new_dw->createParticleSubset(numParticles, matl, patch);
      }
      
      new_dw->allocateAndPut(*pvar, label, psubset);

      vector<ParticleSubset*> subsets;      
      vector<ParticleVariableBase*> srcs;
      for (list<const Patch*>::iterator iter = oldPatches.begin();
	   iter != oldPatches.end(); iter++) {
	ParticleSubset* oldPsubset = dw_->getParticleSubset(matl, *iter);
	srcs.push_back(dw_->getParticleVariable(label, oldPsubset));
	subsets.push_back(oldPsubset);
      }
      pvar->gather(psubset, subsets, srcs);
      //cerr << "Putting " << label->getName() << " for patch " << patch << " matl " << matl << endl;
      //new_dw->put(var, label, matl, patch);
      delete var; // a clone should have been put in the dw      
    }
  }
}

