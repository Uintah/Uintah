#ifndef UINTAH_HOMEBREW_MAPPER_H
#define UINTAH_HOMEBREW_MAPPER_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <string>

namespace Uintah {
    class Task;
    class VarLabel;
    class ProcessorContext;
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
       Scheduler();
       virtual ~Scheduler();
       
       //////////
       // Insert Documentation Here:
       virtual void initialize() = 0;
       
       //////////
       // Insert Documentation Here:
       virtual void execute(const ProcessorContext * pc, 
			          DataWarehouseP   & dwp ) = 0;
       
       //////////
       // Insert Documentation Here:
       virtual void addTask(Task* t) = 0;
       
       //////////
       // Insert Documentation Here:
       virtual DataWarehouseP createDataWarehouse( int generation ) = 0;
       
    private:
       Scheduler(const Scheduler&);
       Scheduler& operator=(const Scheduler&);
    };
    
} // end namespace Uintah

//
// $Log$
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
