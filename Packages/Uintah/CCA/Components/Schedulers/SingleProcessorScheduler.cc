
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler(const ProcessorGroup* myworld,
    	    	    	    	    	    	   Output* oport)
   : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

void
SingleProcessorScheduler::execute(const ProcessorGroup * pg)
{
  ASSERT(dt != 0);
  int ntasks = dt->numTasks();
  if(ntasks == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  dbg << "Executing " << ntasks << " tasks\n";

  makeTaskGraphDoc(dt);

  for(int i=0;i<ntasks;i++){
    double start = Time::currentSeconds();
    DetailedTask* task = dt->getTask(i);
    task->doit(pg, dw[Task::OldDW], dw[Task::NewDW]);
    double dt = Time::currentSeconds()-start;
    dbg << "Completed task: " << task->getTask()->getName()
	<< " (" << dt << " seconds)\n";
    scrub(task);
    emitNode(task, start, dt);
  }

  dw[1]->finalize();
  finalizeNodes();
}

void
SingleProcessorScheduler::scheduleParticleRelocation(const LevelP& level,
						     const VarLabel* old_posLabel,
						     const vector<vector<const VarLabel*> >& old_labels,
						     const VarLabel* new_posLabel,
						     const vector<vector<const VarLabel*> >& new_labels,
						     const MaterialSet* matls)
{
  reloc.scheduleParticleRelocation(this, d_myworld, 0, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels, matls);
}
