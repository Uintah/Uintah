#include <Packages/Uintah/CCA/Components/PatchCombiner/UdaReducer.h>
#include <Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/MaterialSetP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>

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

void UdaReducer::problemSetup(const ProblemSpecP& /*params*/, GridP& grid,
				 SimulationStateP& state)
{
  d_sharedState = state;

  
  dataArchive_ = scinew DataArchive(udaDir_, d_myworld->myrank(), d_myworld->size());
  dataArchive_->queryTimesteps(timesteps_, times_);
  dataArchive_->turnOffXMLCaching();

  // try to add a time to times_ that will get us passed the end of the
  // simulation -- yes this is a hack!!
  times_.push_back(10 * times_[times_.size() - 1]);
  timesteps_.push_back(999999999);

  oldGrid_ = dataArchive_->queryGrid(times_[0]);
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
UdaReducer::scheduleTimeAdvance( const LevelP& level,
				    SchedulerP& sched, int, int )
{
  
  double time = times_[timeIndex_];
  const PatchSet* eachPatch = level->eachPatch();
    
  // so we can tell the task which matls to use (sharedState doesn't have
  // any matls defined, so we can't use that).
  MaterialSetP allMatls = scinew MaterialSet();
  allMatls->createEmptySubsets(1);
  MaterialSubset* allMS = allMatls->getSubset(0);

  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;
  Task* t = scinew Task("UdaReducer::readAndSetDelT", this, &UdaReducer::readAndSetDelT, sched.get_rep());
  for (unsigned int i = 0; i < labels_.size(); i++) {
    VarLabel* label = labels_[i];

    ConsecutiveRangeSet matlsRangeSet;
    for (int i = 0; i < eachPatch->size(); i++) {
      const Patch* patch = eachPatch->getSubset(i)->get(0);
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

    t->computes(label, matls->getUnion());
  }

  MaterialSubsetP globalMatl = scinew MaterialSubset();
  globalMatl->add(-1);
  t->computes(delt_label, globalMatl.get_rep());
  sched->addTask(t, lb->createPerProcessorPatchSet(level), allMatls.get_rep());
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


void UdaReducer::readAndSetDelT(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw,
				   Scheduler* /*sched*/)
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

  //cerr << "deleted and recreated dw\n";
  double delt;
  dataArchive_->restartInitialize(timestep, oldGrid_, new_dw, lb, &time, &delt);
  //cerr << "restartInitialize done\n";
  
  // don't use that delt -- jump to the next output timestep
  delt = times_[timeIndex_] - time;
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
