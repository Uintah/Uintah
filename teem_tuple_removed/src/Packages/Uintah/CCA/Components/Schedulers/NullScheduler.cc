

#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("NullScheduler", false);

NullScheduler::NullScheduler(const ProcessorGroup* myworld,
			     Output* oport)
  : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
  delt = VarLabel::create("delT",
			  ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
  firstTime=true;
}

NullScheduler::~NullScheduler()
{
}

SchedulerP
NullScheduler::createSubScheduler()
{
  NullScheduler* newsched = new NullScheduler(d_myworld, m_outPort);
  UintahParallelPort* lbp = getPort("load balancer");
  newsched->attachPort("load balancer", lbp);
  return newsched;
}

void
NullScheduler::verifyChecksum()
{
  // Not used in NullScheduler
}

void 
NullScheduler::advanceDataWarehouse(const GridP& grid)
{
  for(int i=0;i<(int)dws.size();i++)
    if( !dws[i] )
      dws[i] = scinew OnDemandDataWarehouse(d_myworld, this, 0, grid);
}

void
NullScheduler::actuallyCompile()
{
  if( dts_ )
    delete dts_;
  if(graph.getNumTasks() == 0){
    dts_=0;
    return;
  }

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  dts_ = graph.createDetailedTasks(lb, useInternalDeps() );

  if(dts_->numTasks() == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  
  lb->assignResources(*dts_, d_myworld);

  graph.createDetailedDependencies(dts_, lb);
  releasePort("load balancer");

  dts_->assignMessageTags(d_myworld->myrank());
}

void
NullScheduler::execute()
{
  if(dts_ == 0){
    cerr << "NullScheduler skipping execute, no tasks\n";
    return;
  }
  if(firstTime){
    firstTime=false;
    dws[dws.size()-1]->put(delt_vartype(1.0), delt);
  }
}

void
NullScheduler::scheduleParticleRelocation(const LevelP&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel* /*particleIDLabel*/,
					  const MaterialSet*)
{
}
