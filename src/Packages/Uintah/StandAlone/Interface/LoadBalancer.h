
#ifndef UINTAH_HOMEBREW_LOADBALANCER_H
#define UINTAH_HOMEBREW_LOADBALANCER_H

#include <Uintah/Parallel/UintahParallelPort.h>

namespace Uintah {
   class ProcessorGroup;
   class TaskGraph;
/**************************************

CLASS
   LoadBalancer
   
   Short description...

GENERAL INFORMATION

   LoadBalancer.h

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

    class LoadBalancer : public UintahParallelPort {
    public:
       LoadBalancer();
       virtual ~LoadBalancer();

       virtual void assignResources(TaskGraph& tg, const ProcessorGroup* resources) = 0;

    private:
       LoadBalancer(const LoadBalancer&);
       LoadBalancer& operator=(const LoadBalancer&);
    };
    
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/06/17 07:06:46  sparker
// Changed ProcessorContext to ProcessorGroup
//
//

#endif
