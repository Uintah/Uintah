#ifndef UINTAH_HOMEBREW_MPISCHEDULER_H
#define UINTAH_HOMEBREW_MPISCHEDULER_H

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
   MPIScheduler
   
   Short description...

GENERAL INFORMATION

   MPIScheduler.h

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

   class MPIScheduler : public UintahParallelComponent, public Scheduler {
   public:
      MPIScheduler(const ProcessorGroup* myworld, Output* oport);
      virtual ~MPIScheduler();
      
      //////////
      // Insert Documentation Here:
      virtual void initialize();
      
      //////////
      // Insert Documentation Here:
      virtual void execute( const ProcessorGroup * pc,
			          DataWarehouseP   & dwp );
      
      //////////
      // Insert Documentation Here:
      virtual void addTask(Task* t);
      
      //////////
      // Insert Documentation Here:
      virtual DataWarehouseP createDataWarehouse( int generation );
      
   private:
      MPIScheduler(const MPIScheduler&);
      MPIScheduler& operator=(const MPIScheduler&);

      TaskGraph graph;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/07/26 20:14:11  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.2  2000/06/17 07:04:54  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
//

#endif
