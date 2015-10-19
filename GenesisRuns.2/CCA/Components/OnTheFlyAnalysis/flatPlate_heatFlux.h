/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_flatPlate_heatFlux_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_flatPlate_heatFlux_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   flatPlate_heatFlux
   
GENERAL INFORMATION

   flatPlate_heatFlux.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   flatplate_heatFlux

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class flatPlate_heatFlux : public AnalysisModule {
  public:
    flatPlate_heatFlux(ProblemSpecP& prob_spec,
              SimulationStateP& sharedState,
		Output* dataArchiver);
              
    flatPlate_heatFlux();
                    
    virtual ~flatPlate_heatFlux();
   
    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP& sharedState);
                              
                              
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void restartInitialize();
                                    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);
   
    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level) {};
                                      
  private:

    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
                    
    void doAnalysis(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
                    
    
    // general labels
    class total_heatRateLabel {
    public:
      VarLabel* total_heatRateLabel;
    };
    
    total_heatRateLabel* v_lb;
    MPMLabel* M_lb;
       
    struct plane{    // plane geometry
      Point startPt;
      Point endPt; 
    };   
       
    //__________________________________
    // global constants
    SimulationStateP d_sharedState;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    
    const Material* d_matl;
    MaterialSet* d_matl_set;
    const MaterialSubset* d_matl_sub;
    std::vector<plane*> d_plane;
    Vector d_oneOrZero;
    Point d_corner_pt[4];
  };
}

#endif
