
// $Id$

#include <Uintah/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
using SCICore::Thread::Time;

static SCICore::Util::DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler(const ProcessorGroup* myworld,
    	    	    	    	    	    	   Output* oport)
   : UintahParallelComponent(myworld), Scheduler(oport)
{
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
}

void
SingleProcessorScheduler::initialize()
{
   graph.initialize();
}

void
SingleProcessorScheduler::execute(const ProcessorGroup * pc,
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

   emitEdges(tasks);

   for(int i=0;i<ntasks;i++){
      time_t t = time(NULL);
      double start = Time::currentSeconds();
      tasks[i]->doit(pc);
      double dt = Time::currentSeconds()-start;
      dbg << "Completed task: " << tasks[i]->getName();
      if(tasks[i]->getPatch())
	 dbg << " on patch " << tasks[i]->getPatch()->getID();
      dbg << " (" << dt << " seconds)\n";

      emitNode(tasks[i], t, dt);
   }

   dw->finalize();
   finalizeNodes();
}

void
SingleProcessorScheduler::addTask(Task* task)
{
   graph.addTask(task);
}

DataWarehouseP
SingleProcessorScheduler::createDataWarehouse( int generation )
{
    return scinew OnDemandDataWarehouse(d_myworld, generation );
}

//
// $Log$
// Revision 1.6  2000/07/26 20:14:11  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.5  2000/07/25 20:59:28  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.4  2000/07/19 21:47:59  jehall
// - Changed task graph output to XML format for future extensibility
// - Added statistical information about tasks to task graph output
//
// Revision 1.3  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.2  2000/06/16 22:59:39  guilkey
// Expanded "cycle detected" print statement
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.20  2000/06/15 21:57:11  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.19  2000/06/14 23:43:25  jehall
// - Made "cycle detected" exception more informative
//
// Revision 1.18  2000/06/08 17:11:39  jehall
// - Added quotes around task names so names with spaces are parsable
//
// Revision 1.17  2000/06/03 05:27:23  sparker
// Fixed dependency analysis for reduction variables
// Removed warnings
// Now allow for task patch to be null
// Changed DataWarehouse emit code
//
// Revision 1.16  2000/05/30 20:19:22  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.15  2000/05/30 17:09:37  dav
// MPI stuff
//
// Revision 1.14  2000/05/21 20:10:48  sparker
// Fixed memory leak
// Added scinew to help trace down memory leak
// Commented out ghost cell logic to speed up code until the gc stuff
//    actually works
//
// Revision 1.13  2000/05/19 18:35:09  jehall
// - Added code to dump the task dependencies to a file, which can be made
//   into a pretty dependency graph.
//
// Revision 1.12  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.11  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.10  2000/05/05 06:42:42  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.9  2000/04/28 21:12:04  jas
// Added some includes to get it to compile on linux.
//
// Revision 1.8  2000/04/26 06:48:32  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/20 18:56:25  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/19 21:20:02  dav
// more MPI stuff
//
// Revision 1.5  2000/04/19 05:26:10  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.4  2000/04/11 07:10:40  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 01:03:16  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
