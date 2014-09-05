#ifndef UINTAH_HOMEBREW_ParticleLoadBalancer_H
#define UINTAH_HOMEBREW_ParticleLoadBalancer_H

#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <set>

namespace Uintah {
   /**************************************
     
     CLASS
       ParticleLoadBalancer
      
       Short Description...
      
     GENERAL INFORMATION
      
       ParticleLoadBalancer.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       ParticleLoadBalancer
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class ParticleLoadBalancer : public LoadBalancer, public UintahParallelComponent {
   public:
     ParticleLoadBalancer(const ProcessorGroup* myworld);
     ~ParticleLoadBalancer();
     virtual void assignResources(DetailedTasks& tg, const ProcessorGroup*);
     virtual int getPatchwiseProcessorAssignment(const Patch* patch,
						  const ProcessorGroup* resources);
    virtual int getOldProcessorAssignment(const VarLabel* var,
					  const Patch* patch, const int matl, 
					  const ProcessorGroup* pg); 
    virtual bool needRecompile(double time, double delt, const GridP& grid); 
    virtual void problemSetup(ProblemSpecP& pspec);
     virtual void createNeighborhood(const GridP& grid, const ProcessorGroup*,
				    const Scheduler*);
     virtual bool inNeighborhood(const PatchSubset*, const MaterialSubset*);
     virtual bool inNeighborhood(const Patch*);

     virtual const PatchSet* createPerProcessorPatchSet(const LevelP& level,
							const ProcessorGroup* resources);
   private:
     ParticleLoadBalancer(const ParticleLoadBalancer&);
     ParticleLoadBalancer& operator=(const ParticleLoadBalancer&);

     void assignPatches(const LevelP& level, const ProcessorGroup*,
			const Scheduler* sch);
     void assignPatches2(const LevelP& level, const ProcessorGroup*,
			const Scheduler* sch);

     std::set<const Patch*> d_neighbors;
     std::vector<int> d_processorAssignment;
     std::vector<int> d_oldAssignment;

     double d_lbInterval;
     double d_currentTime;
     double d_lastLbTime;

     int d_lbTimestepInterval;
     int d_currentTimestep;
     int d_lastLbTimestep;

     bool d_do_AMR;
     ProblemSpecP d_pspec;

     enum {
       idle = 0, postLoadBalance = 1, needLoadBalance = 2
     };

     int d_state; //< either idle, needLoadBalance, postLoadBalance
   };
} // End namespace Uintah


#endif

