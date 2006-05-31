
#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_lineExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_lineExtract_h
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
   lineExtract
   
GENERAL INFORMATION

   lineExtract.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   lineExtract

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class lineExtract : public AnalysisModule {
  public:
    lineExtract(ProblemSpecP& prob_spec,
                    SimulationStateP& sharedState,
		      Output* dataArchiver);
    lineExtract();
                    
    virtual ~lineExtract();
   
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
                    
    void createFile(string& filename);
    
    void createDirectory(string& lineName, string& levelIndex);
                    
    
    // general labels
    class lineExtractLabel {
    public:
      VarLabel* lastWriteTimeLabel;
    };
    
    
    
    lineExtractLabel* ps_lb;
   

    struct line{
      string name;  
      Point startPt;
      Point endPt;
      int loopDir;    // direction to loop over
    };
    
    
       
    //__________________________________
    // global constants
    double d_writeFreq; 
    double d_StartTime;
    double d_StopTime;
    vector<VarLabel*> d_varLabels;
    SimulationStateP d_sharedState;
    vector<line*> d_lines;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    const Material* d_matl;
    MaterialSet* d_matl_set;
    
  
  };
}

#endif
