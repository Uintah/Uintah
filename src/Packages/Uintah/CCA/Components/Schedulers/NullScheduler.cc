

#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
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
   delt = scinew VarLabel("delT",
    ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   firstTime=true;
}

NullScheduler::~NullScheduler()
{
}

void 
NullScheduler::advanceDataWarehouse(const GridP& grid)
{
  if(!dw[1])
    dw[1]=scinew OnDemandDataWarehouse(d_myworld, 0, grid);
}

void
NullScheduler::compile(const ProcessorGroup* pg, bool init_timestep)
{
  if(dt)
    delete dt;
  dt = graph.createDetailedTasks(pg);

  if(dt->numTasks() == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  lb->assignResources(*dt, d_myworld);

  graph.createDetailedDependencies(dt, lb, pg);
  releasePort("load balancer");

  dt->assignMessageTags();
  int me=pg->myrank();
  dt->computeLocalTasks(me);
  dt->createScrublists(init_timestep);
}

void
NullScheduler::execute(const ProcessorGroup *)
{
  ASSERT(dt != 0);
  if(firstTime){
    firstTime=false;
    dw[Task::NewDW]->put(delt_vartype(1.0), delt);
  }
}

void
NullScheduler::scheduleParticleRelocation(const LevelP&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const MaterialSet*)
{
}
