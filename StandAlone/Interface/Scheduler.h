#ifndef UINTAH_HOMEBREW_MAPPER_H
#define UINTAH_HOMEBREW_MAPPER_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <string>

namespace Uintah {

  namespace Grid {
    class Task;
  }

  namespace Parallel {
    class ProcessorContext;
  }

namespace Interface {

using Uintah::Parallel::UintahParallelPort;
using Uintah::Parallel::ProcessorContext;
using Uintah::Grid::Task;

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
    virtual void execute(const ProcessorContext*) = 0;

    //////////
    // Insert Documentation Here:
    virtual void addTarget(const std::string& name) = 0;

    //////////
    // Insert Documentation Here:
    virtual void addTask(Task* t) = 0;

    //////////
    // Insert Documentation Here:
    virtual DataWarehouseP createDataWarehouse() = 0;

private:
    Scheduler(const Scheduler&);
    Scheduler& operator=(const Scheduler&);
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
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
