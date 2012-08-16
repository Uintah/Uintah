/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/PatchCombiner/UdaReducer.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>

#include <iomanip>

using namespace std;
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
                              const ProblemSpecP& /*restart_prob_spec*/, 
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

  //cout << d_myworld->myrank() << "  Calling DA::QuearyMaterials a bunch of times " << endl;

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
					       patch, timeIndex_));
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

    allMS->sort();

    for (int l = 0; l < grid->numLevels(); l++) {
      t->computes(label, grid->getLevel(l)->allPatches()->getUnion(), matls->getUnion());
    }
  }

  //cout << d_myworld->myrank() << "  Done Calling DA::QuearyMaterials a bunch of times " << endl;
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
  //cout << d_myworld->myrank() << "   readAndSetVArs\n";
  double time = times_[timeIndex_];
  //int timestep = timesteps_[timeIndex_];

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
  
  if (d_myworld->myrank() == 0)
    cout << "   Incrementing time " << timeIndex_ << " and time " << time << endl;


  dataArchive_->restartInitialize(timeIndex_, oldGrid_, new_dw, lb, &time);
  timeIndex_++;
}

void UdaReducer::readAndSetDelT(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* new_dw)
  
{
  // don't use the delt produced in restartInitialize.
  double delt = times_[timeIndex_] - times_[timeIndex_-1];
  //cout << "   Putting delt " << delt << endl;
  delt_vartype delt_var = delt;
  new_dw->put(delt_var, delt_label);
  //cout << d_myworld->myrank() << "  Done Calling readAndSetDelt " << endl;
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
  bool recompile = gridChanged;
  gridChanged = false;

  vector<int> newNumMatls(oldGrid_->numLevels());
  for (int i = 0; i < oldGrid_->numLevels(); i++) {
    newNumMatls[i] = dataArchive_->queryNumMaterials(*oldGrid_->getLevel(i)->patchesBegin(), timeIndex_);
    if (i >=(int) numMaterials_.size() || numMaterials_[i] != newNumMatls[i])
      recompile = true;
  }
  numMaterials_ = newNumMatls;
  return recompile;
}

// called by the SimController once per timestep
GridP UdaReducer::getGrid() 
{ 
  GridP newGrid = dataArchive_->queryGrid(timeIndex_);
  
  if (oldGrid_ == 0 || !(*newGrid.get_rep() == *oldGrid_.get_rep())) {
    gridChanged = true;
    if (d_myworld->myrank() == 0) cout << "     NEW GRID!!!!\n";
    oldGrid_ = newGrid;
    lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return oldGrid_; 
}
