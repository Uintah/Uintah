#ifndef UINTAH_HOMEBREW_NirvanaLoadBalancer_H
#define UINTAH_HOMEBREW_NirvanaLoadBalancer_H

#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerCommon.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Core/Geometry/IntVector.h>
#include <set>

namespace Uintah {
  using namespace SCIRun;
   /**************************************
     
     CLASS
       NirvanaLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       NirvanaLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       NirvanaLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
  class NirvanaLoadBalancer : public LoadBalancerCommon {
  public:
    NirvanaLoadBalancer(const ProcessorGroup* myworld, const IntVector& layout);
    ~NirvanaLoadBalancer();
    virtual void assignResources(DetailedTasks& tg);
    virtual void assignResources(DetailedTasks3& tg);
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
   private:
     NirvanaLoadBalancer(const NirvanaLoadBalancer&);
     NirvanaLoadBalancer& operator=(const NirvanaLoadBalancer&);
     IntVector layout;
     int npatches;
     int numhosts;
     int numProcs;
     int processors_per_host;
     int patches_per_processor;
     IntVector d;
   };
} // End namespace Uintah

#endif
