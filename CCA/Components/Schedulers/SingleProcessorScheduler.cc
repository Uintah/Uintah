
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#ifdef USE_PERFEX_COUNTERS
#include "counters.h"
#endif

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("SingleProcessorScheduler", false);

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
  SingleProcessorScheduler* newsched = new SingleProcessorScheduler(d_myworld, m_outPort, this);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
SingleProcessorScheduler::verifyChecksum()
{
  // Not used in SingleProcessorScheduler
}

void
SingleProcessorScheduler::actuallyCompile(const ProcessorGroup* pg)
{
  if(dts_)
    delete dts_;

  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  dts_ = graph.createDetailedTasks( pg, lb, useInternalDeps() );

  lb->assignResources(*dts_, d_myworld);

  if (useInternalDeps()) {
    graph.createDetailedDependencies(dts_, lb, pg);
  }
  
  releasePort("load balancer");
}

void
SingleProcessorScheduler::execute(const ProcessorGroup * pg)
{
  if(dts_ == 0){
    cerr << "SingleProcessorScheduler skipping execute, no tasks\n";
    return;
  }
  int ntasks = dts_->numTasks();
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

  makeTaskGraphDoc( dts_ );

  dts_->initializeScrubs(dws);

  for(int i=0;i<ntasks;i++){
#ifdef USE_PERFEX_COUNTERS
    start_counters(0, 19);  
#endif    
    double start = Time::currentSeconds();
    DetailedTask* task = dts_->getTask( i );
    if(dbg.active())
      dbg << "Running task: " << task->getTask()->getName() << "\n";
    task->doit(pg, dws, plain_old_dws);
    if(dbg.active())
      dbg << "calling done\n";
    task->done(dws);

    double delT = Time::currentSeconds()-start;
    long long flop_count = 0;
#ifdef USE_PERFEX_COUNTERS
    long long dummy;
    read_counters(0, &dummy, 19, &flop_count);
#endif
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
