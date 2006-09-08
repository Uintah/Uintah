#include <Packages/Uintah/CCA/Components/PatchCombiner/UdaReducer.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/MaterialSetP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

#include <iomanip>

using namespace Uintah;

UdaReducer::UdaReducer(const ProcessorGroup* world, string udaDir)
  : UintahParallelComponent(world), udaDir_(udaDir), dataArchive_(0),
    timeIndex_(0)
{
  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());
}

UdaReducer::~UdaReducer()
{
  delete dataArchive_;
  
  VarLabel::destroy(delt_label);
  for (unsigned int i = 0; i < labels_.size(); i++)
    VarLabel::destroy(labels_[i]);
}

void UdaReducer::problemSetup(const ProblemSpecP& /*params*/, 
                              const ProblemSpecP& /*materials_ps*/, 
                              GridP& grid, SimulationStateP& state)
{
  d_sharedState = state;

  
  dataArchive_ = scinew DataArchive(udaDir_, d_myworld->myrank(), d_myworld->size());
  dataArchive_->queryTimesteps(timesteps_, times_);
  dataArchive_->turnOffXMLCaching();

  // try to add a time to times_ that will get us passed the end of the
  // simulation -- yes this is a hack!!
  times_.push_back(10 * times_[times_.size() - 1]);
  timesteps_.push_back(999999999);

  //oldGrid_ = dataArchive_->queryGrid(times_[0]);
}

void UdaReducer::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  // labels_ should be empty, but just in case...
  for (unsigned int i = 0; i < labels_.size(); i++)
    VarLabel::destroy(labels_[i]);
  labels_.clear();

  lb = sched->getLoadBalancer();

  vector<string> names;
  vector< const TypeDescription *> typeDescriptions;
  dataArchive_->queryVariables(names, typeDescriptions);
  for (unsigned int i = 0; i < names.size(); i++) {
    labels_.push_back(VarLabel::create(names[i], typeDescriptions[i]));
  }

  Task* t = scinew Task("UdaReducer::initialize", this, &UdaReducer::initialize);
  t->computes(delt_label);
  MaterialSetP globalMatlSet = scinew MaterialSet();
  globalMatlSet->add(-1);
  sched->addTask(t, level->eachPatch() /* ??? -- what to do for multi-proc */,
		 globalMatlSet.get_rep());
}

void
UdaReducer::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{

  // should only get called once from SimCntrl, independent of #levels
  GridP grid = level->getGrid();

  const PatchSet* perProcPatches = lb->getPerProcessorPatchSet(grid);

  cout << "   STA: passed level: " << level.get_rep() << "  stored level " << oldGrid_->getLevel(0).get_rep() << endl;

  double time = times_[timeIndex_];

  // so we can tell the task which matls to use (sharedState doesn't have
  // any matls defined, so we can't use that).
  MaterialSetP allMatls = scinew MaterialSet();
  allMatls->createEmptySubsets(1);
  MaterialSubset* allMS = allMatls->getSubset(0);

  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;
  Task* t = scinew Task("UdaReducer::readAndSetVars", this, &UdaReducer::readAndSetVars);
  for (unsigned int i = 0; i < labels_.size(); i++) {
    VarLabel* label = labels_[i];

    ConsecutiveRangeSet matlsRangeSet;
    for (int i = 0; i < perProcPatches->getSubset(d_myworld->myrank())->size(); i++) {
      const Patch* patch = perProcPatches->getSubset(d_myworld->myrank())->get(i);
      matlsRangeSet = matlsRangeSet.
        unioned(dataArchive_->queryMaterials(label->getName(), 
					       patch, time));
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
        if (!allMS->contains(*iter))
          allMS->add(*iter);
	matls_vec.push_back(*iter);
      }
      matls->addAll(matls_vec);
      prevRangeSet = matlsRangeSet;
      prevMatlSet = matls;
    }

    for (int l = 0; l < grid->numLevels(); l++)
      t->computes(label, grid->getLevel(l)->allPatches()->getUnion(), matls->getUnion());
  }

  MaterialSubsetP globalMatl = scinew MaterialSubset();
  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls.get_rep());

  Task* t2 = scinew Task("UdaReducer::readAndSetDelt", this, &UdaReducer::readAndSetDelT);
  globalMatl->add(-1);
  t2->computes(delt_label, grid->getLevel(0).get_rep(), globalMatl.get_rep());
  sched->addTask(t2, perProcPatches, allMatls.get_rep());
} // end scheduleTimeAdvance()

void UdaReducer::initialize(const ProcessorGroup*,
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


void UdaReducer::readAndSetVars(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* new_dw)
{
  double time = times_[timeIndex_];
  int timestep = timesteps_[timeIndex_];

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
  
  timeIndex_++;

  cout << "   Incrementing time " << timeIndex_ << " and time " << time << endl;

  //cerr << "deleted and recreated dw\n";
  dataArchive_->restartInitialize(timestep, oldGrid_, new_dw, lb, &time);
  //cerr << "restartInitialize done\n";
}

void UdaReducer::readAndSetDelT(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* new_dw)
  
{
  // don't use the delt produced in restartInitialize.
  double delt = times_[timeIndex_] - times_[timeIndex_-1];
  cout << "   Putting delt " << delt << endl;
  delt_vartype delt_var = delt;
  new_dw->put(delt_var, delt_label);
  //cerr << "done with reading\n";
}

double UdaReducer::getMaxTime()
{
  if (times_.size() <= 1)
    return 0;
  else
    return times_[times_.size()-2]; // the last one is the hacked one, see problemSetup
}

bool UdaReducer::needRecompile(double time, double dt,
                               const GridP& grid)
{
  bool recompile = false;
  GridP newGrid = dataArchive_->queryGrid(times_[timeIndex_]);
  if (newGrid != oldGrid_) {
    oldGrid_ = newGrid;
    lb->possiblyDynamicallyReallocate(newGrid, true);
    recompile =  true;
  }

  vector<int> newNumMatls(newGrid->numLevels());
  for (int i = 0; i < newGrid->numLevels(); i++) {
    newNumMatls[i] = dataArchive_->queryNumMaterials(*newGrid->getLevel(i)->patchesBegin(), times_[timeIndex_]);
    if (i >=(int) numMaterials_.size() || numMaterials_[i] != newNumMatls[i])
      recompile = true;
  }
  numMaterials_ = newNumMatls;
  return recompile;
}
