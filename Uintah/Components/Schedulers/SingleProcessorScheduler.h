#ifndef UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H
#define UINTAH_HOMEBREW_SinglePROCESSORSCHEDULER_H

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Components/Schedulers/TaskGraph.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/TaskProduct.h>
#include <Uintah/Grid/Task.h>
#include <vector>
#include <map>

namespace Uintah {
   class Task;
/**************************************

CLASS
   SingleProcessorScheduler
   
   Short description...

GENERAL INFORMATION

   SingleProcessorScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Scheduler_Brain_Damaged

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SingleProcessorScheduler : public UintahParallelComponent, public Scheduler {
   public:
      SingleProcessorScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~SingleProcessorScheduler();
      
      //////////
      // Insert Documentation Here:
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void execute( const ProcessorGroup * pc,
			          DataWarehouseP   & old_dwp,
			          DataWarehouseP   & dwp );
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);
      
      //////////
      // Insert Documentation Here:
      virtual DataWarehouseP createDataWarehouse(DataWarehouseP& parent_dw);
      
      //////////
      // Insert Documentation Here:
      virtual void scheduleParticleRelocation(const LevelP& level,
					      DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw,
					      const VarLabel* old_posLabel,
					      const vector<const VarLabel*>& old_labels,
					      const VarLabel* new_posLabel,
					      const vector<const VarLabel*>& new_labels,
					      int numMatls);
   private:
      const VarLabel* reloc_old_posLabel;
      std::vector<const VarLabel*> reloc_old_labels;
      const VarLabel* reloc_new_posLabel;
      std::vector<const VarLabel*> reloc_new_labels;
      int reloc_numMatls;

      void scatterParticles(const ProcessorGroup*,
			    const Patch* patch,
			    DataWarehouseP& old_dw,
			    DataWarehouseP& new_dw);
      void gatherParticles(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw);
      const VarLabel* scatterGatherVariable;
      SingleProcessorScheduler(const SingleProcessorScheduler&);
      SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);

      TaskGraph graph;
      // id of datawarehouse
      int d_generation;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/07/28 03:01:54  rawat
// modified createDatawarehouse and added getTop()
//
// Revision 1.4  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.3  2000/07/26 20:14:11  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.2  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.11  2000/05/19 18:35:10  jehall
// - Added code to dump the task dependencies to a file, which can be made
//   into a pretty dependency graph.
//
// Revision 1.10  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.9  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.8  2000/05/05 06:42:43  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.7  2000/04/26 06:48:32  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.5  2000/04/19 21:20:02  dav
// more MPI stuff
//
// Revision 1.4  2000/04/19 05:26:10  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/03/17 01:03:16  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//

#endif
