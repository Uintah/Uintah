
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
    	    	    	    	    	    	   Output* oport)
   : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

SchedulerP
SingleProcessorScheduler::createSubScheduler()
{
  SingleProcessorScheduler* newsched = new SingleProcessorScheduler(d_myworld, m_outPort);
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
SingleProcessorScheduler::compile(const ProcessorGroup* pg, bool init_timestep)
{
  if(dts_)
    delete dts_;

  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  if( useInternalDeps() ) {
    dts_ = graph.createDetailedTasks( pg, lb, true );
  }
  else {
    dts_ = graph.createDetailedTasks( pg, lb, false );
  }

  lb->assignResources(*dts_, d_myworld);

  if (useInternalDeps()) {
    graph.createDetailedDependencies(dts_, lb, pg);
  }
  
  releasePort("load balancer");
  dts_->computeLocalTasks(pg->myrank());
  dts_->createScrublists(init_timestep);
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
  dbg << "Executing " << ntasks << " tasks\n";

  makeTaskGraphDoc( dts_ );

  for(int i=0;i<ntasks;i++){
#ifdef USE_PERFEX_COUNTERS
    start_counters(0, 19);  
#endif    
    double start = Time::currentSeconds();
    DetailedTask* task = dts_->getTask( i );
    task->doit(pg, dws_[Task::OldDW], dws_[Task::NewDW]);
    double delT = Time::currentSeconds()-start;
    long long flop_count = 0;
#ifdef USE_PERFEX_COUNTERS
    long long dummy;
    read_counters(0, &dummy, 19, &flop_count);
#endif
    dbg << "Completed task: " << task->getTask()->getName()
	<< " (" << delT << " seconds)\n";
    scrub(task);
    emitNode( task, start, delT, delT, flop_count );
  }

  dws_[ Task::NewDW ]->finalize();
  finalizeNodes();
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
  reloc.scheduleParticleRelocation(this, d_myworld, 0, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels,
				   particleIDLabel, matls);
}
