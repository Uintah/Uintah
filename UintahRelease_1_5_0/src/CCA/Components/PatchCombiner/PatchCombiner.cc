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

#include <CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>

using namespace std;
using namespace Uintah;

PatchCombiner::PatchCombiner(const ProcessorGroup* world, string udaDir)
  : UintahParallelComponent(world), udaDir_(udaDir), dataArchive_(0),
    world_(world), timeIndex_(0)
{
  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());
}

PatchCombiner::~PatchCombiner()
{
  delete dataArchive_;
  
  VarLabel::destroy(delt_label);
  for (unsigned int i = 0; i < labels_.size(); i++)
    VarLabel::destroy(labels_[i]);
}

void PatchCombiner::problemSetup(const ProblemSpecP& /*params*/, 
                                 const ProblemSpecP& /*restart_prob_spec*/, 
                                 GridP& grid, SimulationStateP& state)
{
  d_sharedState = state;

  
  if (world_->size() > 1) {
    throw InternalError("combine_patches not yet supported in parallel", __FILE__, __LINE__);
  }  
  dataArchive_ = scinew DataArchive(udaDir_, world_->myrank(), world_->size());
  dataArchive_->queryTimesteps(timesteps_, times_);
  dataArchive_->turnOffXMLCaching();

  // try to add a time to times_ that will get us passed the end of the
  // simulation -- yes this is a hack!!
  times_.push_back(10 * times_[times_.size() - 1]);
  timesteps_.push_back(999999999);

  GridP newGrid = scinew Grid();
  GridP oldGrid = dataArchive_->queryGrid(0);


  // use a subscheduler cuz the DW we will read data from will have a different
  // grid from the parent scheduler/grid.
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  d_subsched = sched->createSubScheduler();
  d_subsched->initialize(1,1);
  d_subsched->setRestartable(true); 
  d_subsched->clearMappings();
  d_subsched->mapDataWarehouse(Task::OldDW, 0);
  d_subsched->mapDataWarehouse(Task::NewDW, 1);

  const SuperPatchContainer* superPatches;
  for (int i = 0; i < oldGrid->numLevels(); i++) {
    LevelP level = oldGrid->getLevel(i);
    LocallyComputedPatchVarMap patchGrouper;
    const PatchSubset* patches = level->allPatches()->getUnion();
    patchGrouper.addComputedPatchSet(patches);
    patchGrouper.makeGroups();
    superPatches = patchGrouper.getSuperPatches(level.get_rep());
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
	inLow = Min(inLow, patch->getCellLowIndex());
	inHigh = Max(inHigh, patch->getCellHighIndex());
      }
      
      Patch* newPatch =
	newLevel->addPatch(low, high, inLow, inHigh, newGrid.get_rep());
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
PatchCombiner::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  
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
					       oldPatch, timeIndex_));
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
    // require delt just to set up the dependency (what it really needs is new dw).
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

  if (timeIndex_ >= (int) (times_.size()-1)) {
    // error situation - we have run out of timesteps in the uda, but 
    // the time does not satisfy the maxTime, so the simulation wants to 
    // keep going
    cerr << "The timesteps in the uda directory do not extend to the maxTime\n"
         << "in the input.ups file.  To not get this exception, adjust the\n"
         << "maxTime in <udadir>/input.xml to be\n"
         << "between " << (times_.size() >= 3 ? times_[times_.size()-3] : 0)
         << " and " << times_[times_.size()-2] << " (the last time in the uda)\n"
         << "This is not a critical error - it just adds one more timestep\n"
         << "that you may have to remove manually\n\n";
  }
  
  d_subsched->advanceDataWarehouse(oldGrid_);
  //cerr << "deleted and recreated dw\n";
  double delt;
  dataArchive_->restartInitialize(timeIndex_, oldGrid_, d_subsched->get_dw(1), NULL, &time);
  //cerr << "restartInitialize done\n";
  
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
  // dependency for the new dw -- get it so it doesn't complain about requiring
  // something not used.
  delt_vartype delt_var;  
  new_dw->get(delt_var, delt_label);
	      
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    // fill in the new patch with each of the old patches that made it up.
    list<const Patch*>& oldPatches = new2OldPatchMap_[patch];
    PatchSubset* sub = scinew PatchSubset;
    for (list<const Patch*>::iterator iter = oldPatches.begin(); iter != oldPatches.end(); iter++) {
      sub->add(*iter);
    }
    new_dw->transferFrom(d_subsched->get_dw(1), label, sub, matls); 

    delete sub; 
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
  // dependency for the new dw -- get it so it doesn't complain about requiring
  // something not used.
  delt_vartype delt_var;  
  new_dw->get(delt_var, delt_label);
  DataWarehouse* dw = d_subsched->get_dw(1);

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
	numParticles += dw->getParticleSubset(matl, *iter)->numParticles();
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
	ParticleSubset* oldPsubset = dw->getParticleSubset(matl, *iter);
	srcs.push_back(dw->getParticleVariable(label, oldPsubset));
	subsets.push_back(oldPsubset);
      }
      pvar->gather(psubset, subsets, srcs);
      //cerr << "Putting " << label->getName() << " for patch " << patch << " matl " << matl << endl;
      //new_dw->put(var, label, matl, patch);
      delete var; // a clone should have been put in the dw      
    }
  }
}

double PatchCombiner::getMaxTime()
{
  if (times_.size() <= 1)
    return 0;
  else
    return times_[times_.size()-2]; // the last one is the hacked one, see problemSetup
}
