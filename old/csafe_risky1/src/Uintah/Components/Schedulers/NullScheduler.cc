
// $Id$

#include <Uintah/Components/Schedulers/NullScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ScatterGatherBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Util/DebugStream.h>
#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using SCICore::Thread::Time;

static SCICore::Util::DebugStream dbg("NullScheduler", false);

NullScheduler::NullScheduler(const ProcessorGroup* myworld,
    	    	    	    	    	    	   Output* oport)
   : UintahParallelComponent(myworld), Scheduler(oport)
{
  d_generation = 0;
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

//
// $Log$
// Revision 1.1.2.1  2000/10/10 05:33:12  sparker
// Added NullScheduler (used for optimizing taskgraph construction)
//
//
