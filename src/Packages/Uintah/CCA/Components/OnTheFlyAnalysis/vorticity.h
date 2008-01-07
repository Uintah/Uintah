
#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_vorticity_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_vorticity_h
#include <Packages/Uintah/CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   vorticity
   
GENERAL INFORMATION

   vorticity.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   vorticity

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class vorticity : public AnalysisModule {
  public:
    vorticity(ProblemSpecP& prob_spec,
              SimulationStateP& sharedState,
		Output* dataArchiver);
              
    vorticity();
                    
    virtual ~vorticity();
   
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
    class vorticityLabel {
    public:
      VarLabel* vorticityLabel;
    };
    
    vorticityLabel* v_lb;
    ICELabel* I_lb;
       
    //__________________________________
    // global constants
    SimulationStateP d_sharedState;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    const Material* d_matl;
    MaterialSet* d_matl_set;
    const MaterialSubset* d_matl_sub;
  };
}

#endif
