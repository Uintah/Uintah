
#ifndef UINTAH_HOMEBREW_LOADBALANCER_H
#define UINTAH_HOMEBREW_LOADBALANCER_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

namespace Uintah {

  class Patch;
  class ProcessorGroup;
  class DetailedTasks;
  class Scheduler;

/****************************************

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
    
    virtual void assignResources(DetailedTasks& tg,
				 const ProcessorGroup* resources) = 0;
    virtual int getPatchwiseProcessorAssignment(const Patch* patch,
						const ProcessorGroup* resources) = 0;
    virtual int getOldProcessorAssignment(const Patch* patch,
    			          const ProcessorGroup* resources) 
      { return getPatchwiseProcessorAssignment(patch, resources); }
    virtual void createNeighborhood(const GridP& grid, const ProcessorGroup*,
				    const Scheduler* sc) = 0;
    virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*) = 0;
    virtual bool inNeighborhood(const Patch*) = 0;

    virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level,
						       const ProcessorGroup* resources) = 0;

  private:
    LoadBalancer(const LoadBalancer&);
    LoadBalancer& operator=(const LoadBalancer&);
  };
} // End namespace Uintah
    
#endif
