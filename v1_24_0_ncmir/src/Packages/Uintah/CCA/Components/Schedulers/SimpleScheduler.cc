

#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("SimpleScheduler", false);

SimpleScheduler::SimpleScheduler(const ProcessorGroup* myworld,
				 Output* oport) :
  SchedulerCommon( myworld, oport )
{
}

SimpleScheduler::~SimpleScheduler()
{
}

SchedulerP
SimpleScheduler::createSubScheduler()
{
  SimpleScheduler* newsched = new SimpleScheduler(d_myworld, m_outPort);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
SimpleScheduler::verifyChecksum()
{
  // SimpleScheduler doesn't need this
}

void
SimpleScheduler::actuallyCompile()
{
  graph.topologicalSort(tasks);
}

void
SimpleScheduler::execute()
{
  int ntasks = (int)tasks.size();
  if(ntasks == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  dbg << "Executing " << ntasks << " tasks\n";
  
  vector<DataWarehouseP> plain_old_dws(dws.size());
  for(int i=0;i<(int)dws.size();i++)
    plain_old_dws[i] = dws[i].get_rep();
  for(int i=0;i<ntasks;i++){
    double start = Time::currentSeconds();
    Task* task = tasks[i];
    const PatchSet* patchset = task->getPatchSet();
    const MaterialSet* matlset = task->getMaterialSet();
    if(patchset && matlset){
      for(int p=0;p<patchset->size();p++){
	const PatchSubset* patch_subset = patchset->getSubset(p);
	for(int m=0;m<matlset->size();m++){
	  const MaterialSubset* matl_subset = matlset->getSubset(m);
	  task->doit( d_myworld, patch_subset, matl_subset, plain_old_dws);
	}
      }
    } else {
      task->doit(d_myworld, 0, 0, plain_old_dws);
    }
    double dt = Time::currentSeconds()-start;
    dbg << "Completed task: " << tasks[i]->getName()
	<< " (" << dt << " seconds)\n";
  }

  finalizeTimestep();
}

void
SimpleScheduler::scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* old_posLabel,
					    const vector<vector<const VarLabel*> >& old_labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const VarLabel* particleIDLabel,
					    const MaterialSet* matls)
{
  reloc_.scheduleParticleRelocation(this, d_myworld, 0, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels,
				   particleIDLabel, matls);
}
