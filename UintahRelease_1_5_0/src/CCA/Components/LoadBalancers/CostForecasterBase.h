/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_CostForecasterBase_H
#define UINTAH_HOMEBREW_CostForecasterBase_H

#include <Core/Grid/Grid.h>
#include <Core/Parallel/ProcessorGroup.h>
namespace Uintah {
   /**************************************
     
     CLASS
       CostForecasterBase 
      
       Base class for cost forecasters used by the DynamicLoadBalancer.

     GENERAL INFORMATION
      
       CostForecasterBase.h
      
       Justin Luitjens
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
             
     KEYWORDS
       CostForecasterBase
       DynamicLoadBalancer
      
     DESCRIPTION
       This is a base class for cost forecasters.  Forecasters should be 
       able to predict the costs of execution for each patch of the grid.
       The dynamic load balancer will use these costs to help ensure
       an even load balance.

     WARNING
      
     ****************************************/

  class CostForecasterBase {
  public:
    virtual ~CostForecasterBase() {};
    virtual void setMinPatchSize(const vector<IntVector> &min_patch_size) {};
    //add the contribution for region r on level l
    virtual void addContribution(DetailedTask *task, double cost) {};
    //finalize the contributions for this timestep
    virtual void finalizeContributions(const GridP currentGrid) {};
    //compute the weights for all patches in the grid.  Particles are provided in the num_particles vectors.
    virtual void getWeights(const Grid* grid, vector<vector<int> > num_particles, vector<vector<double> >&costs) = 0;
    //sets the decay rate for the exponential average
    virtual void setTimestepWindow(int window) {};
    //initializes the regions in the new level that are not in the old level
    virtual void initializeWeights(const Grid* oldgrid, const Grid* newgrid) {};
    //resets all counters to zero
    virtual void reset() {};
    //returns true if the forecaster is ready to be used
    virtual bool hasData() {return true;}
  private:

  };
} // End namespace Uintah


#endif

