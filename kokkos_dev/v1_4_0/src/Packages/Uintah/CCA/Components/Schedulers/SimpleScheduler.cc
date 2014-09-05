

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
				 Output* oport)
   : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
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
SimpleScheduler::compile(const ProcessorGroup*, bool)
{
  graph.topologicalSort(tasks);
}

void
SimpleScheduler::execute(const ProcessorGroup * pc)
{
  int ntasks = (int)tasks.size();
  if(ntasks == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  dbg << "Executing " << ntasks << " tasks\n";
  
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
	  task->doit( pc, patch_subset, matl_subset, 
		      dws_[Task::OldDW], dws_[Task::NewDW] );
	}
      }
    } else {
      task->doit(pc, 0, 0, dws_[Task::OldDW], dws_[Task::NewDW]);
    }
    double dt = Time::currentSeconds()-start;
    dbg << "Completed task: " << tasks[i]->getName()
	<< " (" << dt << " seconds)\n";
  }

  dws_[Task::NewDW]->finalize();
  finalizeNodes();
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
  reloc.scheduleParticleRelocation(this, d_myworld, 0, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels,
				   particleIDLabel, matls);
}
