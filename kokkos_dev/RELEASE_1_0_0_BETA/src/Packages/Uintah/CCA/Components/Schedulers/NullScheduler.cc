

#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("NullScheduler", false);

NullScheduler::NullScheduler(const ProcessorGroup* myworld,
			     Output* oport)
   : UintahParallelComponent(myworld), Scheduler(oport)
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
NullScheduler::initialize()
{
   graph.initialize();
}

void
NullScheduler::execute(const ProcessorGroup *,
		       DataWarehouseP   & old_dw,
		       DataWarehouseP   & new_dw )
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   lb->assignResources(graph, d_myworld);
   releasePort("load balancer");

   if(firstTime){
      firstTime=false;
      new_dw->put(delt_vartype(1.0), delt);
   }

   new_dw=old_dw;
}

void
NullScheduler::addTask(Task* task)
{
   graph.addTask(task);
}

DataWarehouseP
NullScheduler::createDataWarehouse(DataWarehouseP& parent_dw)
{
  int generation = d_generation++;
  return scinew OnDemandDataWarehouse(d_myworld, generation, parent_dw);
}


void
NullScheduler::scheduleParticleRelocation(const LevelP&,
					  DataWarehouseP&,
					  DataWarehouseP&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  int)
{
}

LoadBalancer*
NullScheduler::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
NullScheduler::releaseLoadBalancer()
{
   releasePort("load balancer");
}

