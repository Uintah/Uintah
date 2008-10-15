#ifndef UINTAH_HOMEBREW_ProfileDriver_H
#define UINTAH_HOMEBREW_ProfileDriver_H

#include <map>
#include <vector>
using namespace std;

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Region.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
namespace Uintah {
   /**************************************
     
     CLASS
       ProfileDriver 
      
       Profiles the execution costs of regions of the domain for use by the 
       DynamicLoadBalancer cost model
      
     GENERAL INFORMATION
      
       ProfileDriver.h
      
       Justin Luitjens
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       ProfileDriver
       DynamicLoadBalancer
      
     DESCRIPTION
       Profiles the execution costs of regions of the domain for use by the 
       DynamicLoadBalancer.  The domain is broken into square patches which
       are use as sample points.  At every timestep the scheduler needs to 
       update the execution costs using addContribution and then finalize the 
       contributions at the end of the timestep using finalizeContribution.

       The DLB uses the getWeights function to assign weights to regions which 
       can then be used to load balance the calculation.
      
     WARNING
      
     ****************************************/

  class ProfileDriver {
    //contribution data point.  
    struct Contribution
    {
      double current; //current contribution that has not been finalized
      double weight;  //current first order weight
      int timestep;  //last timestep this datapoint was updated
      Contribution ()
      {
        current=0;
        weight=0;
        timestep=0;
      }
    };
  public:
    ProfileDriver(const ProcessorGroup* myworld) : d_myworld(myworld), d_timestepWindow(20), timesteps(0) {updateAlpha();};
    void setNumLevels(int num) {costs.resize(num);}
    //add the contribution for region r on level l
    void addContribution(const vector<Region> &regions, const vector<int> &levels, double cost);
    //finalize the contributions for this timestep
    void finalizeContributions();
    //get the contribution for the regions
    void getWeights(int l, const vector<Region> &regions, vector<double> &weights);
    //get the predicted and measured values for each region
    void getMeasuredAndPredictedWeights(int l, const vector<Region> &regions, vector<double> &measured, vector<double> &predicted);
    //sets the decay rate for the exponential average
    void setTimestepWindow(int window) {d_timestepWindow=window, updateAlpha();}
    //initializes the regions in the new level that are not in the old level
    void initializeWeights(const vector<Region> &old_regions, const vector<Region> &new_regions, int level);
    //resets all counters to zero
    void reset();
    //returns true if profiling data exists
    bool hasData() {return timesteps>0;}
  private:
    const void updateAlpha() { d_alpha=2.0/(d_timestepWindow+1); }
    const ProcessorGroup* d_myworld;
            
    int d_timestepWindow;
    double d_alpha;

    vector<map<IntVector, Contribution> > costs;
    int timesteps;
  };
} // End namespace Uintah


#endif

