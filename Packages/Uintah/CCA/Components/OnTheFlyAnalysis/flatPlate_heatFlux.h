
#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_flatPlate_heatFlux_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_flatPlate_heatFlux_h
#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

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
  
   Copyright (C) 2000 SCI Group

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
                              GridP& grid,
                              SimulationStateP& sharedState);
                              
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void restartInitialize();
                                    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);
   
                                      
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
    vector<plane*> d_plane;
    Vector d_oneOrZero;
    Point d_corner_pt[4];
  };
}

#endif
