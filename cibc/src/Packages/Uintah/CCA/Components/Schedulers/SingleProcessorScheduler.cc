
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/TaskGraph.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#ifdef USE_PERFEX_COUNTERS
#include "counters.h"
#endif

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("SingleProcessorScheduler", false);
extern DebugStream taskdbg;

SingleProcessorScheduler::SingleProcessorScheduler(const ProcessorGroup* myworld,
    	    	    	    	    	    	   Output* oport, 
						   SingleProcessorScheduler* parent)
   : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
  m_parent = parent;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

SchedulerP
SingleProcessorScheduler::createSubScheduler()
{
  SingleProcessorScheduler* newsched = scinew SingleProcessorScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
SingleProcessorScheduler::verifyChecksum()
{
  // Not used in SingleProcessorScheduler
}

static
void
printTask( ostream& out, DetailedTask* task )
{
  out << task->getTask()->getName();
  if(task->getPatches()){
    out << " on patches ";
    const PatchSubset* patches = task->getPatches();
    for(int p=0;p<patches->size();p++){
      if(p != 0)
	out << ", ";
      out << patches->get(p)->getID();
    }
  }
}

void
SingleProcessorScheduler::execute(int tgnum /*=0*/, int iteration /*=0*/)
{
  ASSERTRANGE(tgnum, 0, (int)graphs.size());
  TaskGraph* tg = graphs[tgnum];
  tg->setIteration(iteration);
  currentTG_ = tgnum;
  DetailedTasks* dts = tg->getDetailedTasks();

  if (graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(dwmap);
  }

  if(dts == 0){
    cerr << "SingleProcessorScheduler skipping execute, no tasks\n";
    return;
  }
  int ntasks = dts->numTasks();
  if(ntasks == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  ASSERT(dws.size()>=2);
  vector<DataWarehouseP> plain_old_dws(dws.size());
  for(int i=0;i<(int)dws.size();i++)
    plain_old_dws[i] = dws[i].get_rep();
  if(dbg.active()){
    dbg << "Executing " << ntasks << " tasks, ";
    for(int i=0;i<numOldDWs;i++){
      dbg << "from DWs: ";
      if(dws[i])
        dbg << dws[i]->getID() << ", ";
      else
        dbg << "Null, ";
    }
    if(dws.size()-numOldDWs>1){
      dbg << "intermediate DWs: ";
      for(unsigned int i=numOldDWs;i<dws.size()-1;i++)
        dbg << dws[i]->getID() << ", ";
    }
    if(dws[dws.size()-1])
      dbg << " to DW: " << dws[dws.size()-1]->getID();
    else
      dbg << " to DW: Null";
    dbg << "\n";
  }
  
  makeTaskGraphDoc( dts );
  
  dts->initializeScrubs(dws, dwmap);
  
  for(int i=0;i<ntasks;i++){
#ifdef USE_PERFEX_COUNTERS
    start_counters(0, 19);  
#endif    
    double start = Time::currentSeconds();
    DetailedTask* task = dts->getTask( i );
    
    taskdbg << d_myworld->myrank() << " Initiating task: "; printTask(taskdbg, task); taskdbg << '\n';

    printTrackedVars(task, true);
    task->doit(d_myworld, dws, plain_old_dws);
    printTrackedVars(task, false);
    task->done(dws);
    
    taskdbg << d_myworld->myrank() << " Completed task: "; printTask(taskdbg, task); taskdbg << '\n';
    double delT = Time::currentSeconds()-start;
    long long flop_count = 0;
#ifdef USE_PERFEX_COUNTERS
    long long dummy;
    read_counters(0, &dummy, 19, &flop_count);
#endif
    if(dws[dws.size()-1] && dws[dws.size()-1]->timestepAborted()){
      dbg << "Aborting timestep after task: " << *task->getTask() << '\n';
      break;
    }
    if(dbg.active())
      dbg << "Completed task: " << *task->getTask()
          << " (" << delT << " seconds)\n";
    //scrub(task);
    emitNode( task, start, delT, delT, flop_count );
  }
  finalizeTimestep();
}

void
SingleProcessorScheduler::scheduleParticleRelocation(const LevelP& level,
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
