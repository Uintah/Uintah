#ifndef UINTAH_HOMEBREW_ParticleLoadBalancer_H
#define UINTAH_HOMEBREW_ParticleLoadBalancer_H

#include <Packages/Uintah/CCA/Components/Schedulers/LoadBalancerCommon.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <set>
#include <string>

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
    
  class ParticleLoadBalancer : public LoadBalancerCommon {
  public:
    ParticleLoadBalancer(const ProcessorGroup* myworld);
    ~ParticleLoadBalancer();
    virtual int getPatchwiseProcessorAssignment(const Patch* patch);
    virtual int getOldProcessorAssignment(const VarLabel* var,
					  const Patch* patch, const int matl);
    virtual bool needRecompile(double time, double delt, const GridP& grid); 
    virtual bool isDynamic() { return d_dynamicAlgorithm != static_lb; }
    // maintain lb state and call one of the assignPatches functions
    virtual void dynamicReallocation(const GridP& grid, const SchedulerP& sch);
    virtual void restartInitialize(ProblemSpecP& pspec, XMLURL tsurl);
    
  private:
    enum { static_lb, cyclic_lb, random_lb, particle_lb };

    ParticleLoadBalancer(const ParticleLoadBalancer&);
    ParticleLoadBalancer& operator=(const ParticleLoadBalancer&);

    // these functions take care of setting d_processorAssignment on all procs
    // and dynamicReallocation takes care of maintaining the state
    void assignPatchesParticle(const GridP& level, const SchedulerP& sch);
    void assignPatchesRandom(const GridP& level, const SchedulerP& sch);
    void assignPatchesCyclic(const GridP& level, const SchedulerP& sch);

    virtual void setDynamicAlgorithm(std::string algo, double interval, 
                                     int timestepInterval, float cellFactor);
    
    std::vector<int> d_processorAssignment;
    std::vector<int> d_oldAssignment;

    double d_lbInterval;
    double d_lastLbTime;
    
    int d_lbTimestepInterval;
    int d_lastLbTimestep;
    
    bool d_do_AMR;
    ProblemSpecP d_pspec;
    
    enum {
      idle = 0, postLoadBalance = 1, needLoadBalance = 2
    };
    
    float d_cellFactor;
    int d_dynamicAlgorithm;
    int d_particleAlgo;
    int d_state; //< either idle, needLoadBalance, postLoadBalance
  };
} // End namespace Uintah


#endif

