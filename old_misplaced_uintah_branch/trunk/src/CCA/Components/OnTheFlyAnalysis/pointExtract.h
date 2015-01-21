
#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_pointExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_pointExtract_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   pointExtract
   
GENERAL INFORMATION

   pointExtract.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   pointExtract

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class pointExtract : public AnalysisModule {
  public:
    pointExtract(ProblemSpecP& prob_spec,
                    SimulationStateP& sharedState,
		      Output* dataArchiver);
    pointExtract();
                    
    virtual ~pointExtract();
   
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
    
    void parsePoint(const string& stringValue, Point &point);
                    
    void checkForInputError(const string& stringValue, const string& Int_or_float);

    // general labels
    class pointExtractLabel {
    public:
      VarLabel* lastWriteTimeLabel;
    };
    
    
    
    pointExtractLabel* ps_lb;
   

    // pointExtractPoint: a structure to hold
    //   each point and its name
    struct pe_point{
      string name;  
      Point thePt;
    };
    
    
       
    //__________________________________
    // global constants
    double d_writeFreq; 
    double d_StartTime;
    double d_StopTime;
    vector<VarLabel*> d_varLabels;
    SimulationStateP d_sharedState;
    vector<pe_point*> d_points;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    const Material* d_matl;
    MaterialSet* d_matl_set;
    
  
  };
}

#endif
