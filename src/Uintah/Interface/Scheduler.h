
#ifndef UINTAH_HOMEBREW_SCHEDULER_H
#define UINTAH_HOMEBREW_SCHEDULER_H

#include <Uintah/Grid/LevelP.h>
#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/Output.h>
#include <string>
#include <vector>

class DOM_Document;
class DOM_Element;

namespace Uintah {

class Task;
class VarLabel;
class ProcessorGroup;

using std::vector;
using std::string;

class LoadBalancer;
class Task;
class TaskGraph;
class VarLabel;
class ProcessorGroup;

/**************************************

CLASS
   Scheduler
   
   Short description...

GENERAL INFORMATION

   Scheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class Scheduler : public UintahParallelPort {
    public:
       Scheduler(Output* oport);
       virtual ~Scheduler();

       virtual void problemSetup(const ProblemSpecP& prob_spec);

       
       //////////
       // Insert Documentation Here:
       virtual void initialize() = 0;
       
       //////////
       // Insert Documentation Here:
       virtual void execute(const ProcessorGroup * pc, 
			          DataWarehouseP   & old_dwp,
			          DataWarehouseP   & dwp ) = 0;
       
       //////////
       // Insert Documentation Here:
       virtual void addTask(Task* t) = 0;

       virtual LoadBalancer* getLoadBalancer() = 0;
       virtual void releaseLoadBalancer() = 0;
       
       //////////
       // Insert Documentation Here:
       virtual DataWarehouseP createDataWarehouse(DataWarehouseP& parent_dw) = 0;
       //    protected:

       //////////
       // Insert Documentation Here:
       virtual void scheduleParticleRelocation(const LevelP& level,
					       DataWarehouseP& old_dw,
					       DataWarehouseP& new_dw,
					       const VarLabel* posLabel,
					       const vector<vector<const VarLabel*> >& labels,
					       const VarLabel* new_posLabel,
					       const vector<vector<const VarLabel*> >& new_labels,
					       int numMatls) = 0;

    protected:
    	void makeTaskGraphDoc(const vector<Task*>& tasks,
			      bool emit_edges = true);
    	void emitNode(const Task* name, time_t start, double duration);
    	void finalizeNodes(int process=0);
    
    private:
       Scheduler(const Scheduler&);
       Scheduler& operator=(const Scheduler&);

       Output* m_outPort;
       DOM_Document* m_graphDoc;
       DOM_Element* m_nodes;
       //unsigned int m_executeCount;
    };
    
} // end namespace Uintah

//
// $Log$
// Revision 1.21  2000/09/27 00:14:33  witzel
// Changed emitEdges to makeTaskGraphDoc with an option to emit
// the actual edges (only process 0 in the MPI version since all
// process contain the same taskgraph edge information).
//
// Revision 1.20  2000/09/26 21:41:52  dav
// minor formatting
//
// Revision 1.19  2000/09/20 15:50:30  sparker
// Added problemSetup interface to scheduler
// Added ability to get/release the loadBalancer from the scheduler
//   (used for getting processor assignments to create per-processor
//    tasks in arches)
// Added getPatchwiseProcessorAssignment to LoadBalancer interface
//
// Revision 1.18  2000/09/08 17:49:50  witzel
// Changing finalizeNodes so that it outputs different taskgraphs
// in different timestep directories and the taskgraph information
// of different processes in different files.
//
// Revision 1.17  2000/07/28 22:45:17  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.16  2000/07/28 03:01:07  rawat
// modified createDatawarehouse and added getTop function
//
// Revision 1.15  2000/07/27 22:39:53  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.14  2000/07/26 20:14:12  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.13  2000/07/25 20:59:27  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.12  2000/07/19 21:41:52  jehall
// - Added functions for emitting task graph information to reduce redundancy
//
// Revision 1.11  2000/06/17 07:06:47  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.10  2000/06/15 23:14:10  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.9  2000/05/11 20:10:23  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.8  2000/05/05 06:42:46  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.7  2000/04/26 06:49:12  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/19 05:26:19  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/11 07:10:54  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/17 21:02:08  dav
// namespace mods
//
// Revision 1.3  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
