
// $Id$

#include <Uintah/Components/Schedulers/MPIScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
using SCICore::Thread::Time;

static SCICore::Util::DebugStream dbg("MPIScheduler", false);

MPIScheduler::MPIScheduler(const ProcessorGroup* myworld)
   : UintahParallelComponent(myworld)
{
}

MPIScheduler::~MPIScheduler()
{
}

void
MPIScheduler::initialize()
{
   graph.initialize();
}

void
MPIScheduler::execute(const ProcessorGroup * pc,
		      DataWarehouseP   & dw )
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   lb->assignResources(graph, d_myworld);
   releasePort("load balancer");

   vector<Task*> tasks;
   graph.topologicalSort(tasks);

   int ntasks = (int)tasks.size();
   if(ntasks == 0){
      cerr << "WARNING: Scheduler executed, but no tasks\n";
   }
   dbg << "Executing " << ntasks << " tasks\n";
   graph.dumpDependencies();
   for(int i=0;i<ntasks;i++){
      double start = Time::currentSeconds();
      tasks[i]->doit(pc);
      double dt = Time::currentSeconds()-start;
      dbg << "Completed task: " << tasks[i]->getName();
      if(tasks[i]->getPatch())
	 dbg << " on patch " << tasks[i]->getPatch()->getID();
      dbg << " (" << dt << " seconds)\n";
   }

   dw->finalize();
}

void
MPIScheduler::addTask(Task* task)
{
   graph.addTask(task);
}

DataWarehouseP
MPIScheduler::createDataWarehouse( int generation )
{
    return scinew OnDemandDataWarehouse(d_myworld, generation );
}

//
// $Log$
// Revision 1.2  2000/06/17 07:04:53  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
