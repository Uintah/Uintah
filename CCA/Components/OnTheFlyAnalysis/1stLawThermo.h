/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_FirstLawThermo_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_FirstLawThermo_h
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>

#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/ICELabel.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   FirstLawThermo
   
GENERAL INFORMATION

   FirstLawThermo.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   FirstLawThermo

DESCRIPTION
   Perform a first law of thermodynamics control volume analysis
  
WARNING
  
****************************************/
  class FirstLawThermo : public AnalysisModule {
  public:
    FirstLawThermo(ProblemSpecP& prob_spec,
                  SimulationStateP& sharedState,
		    Output* dataArchiver);
              
    FirstLawThermo();
                    
    virtual ~FirstLawThermo();
   
    virtual void problemSetup(const ProblemSpecP& prob_spec,
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
                    
    void computeContributions(const ProcessorGroup* pg,
                              const PatchSubset* patches,
                              const MaterialSubset* matl_sub ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);
                              
    void faceInfo(const std::string fc,
                   Patch::FaceType& face_side, 
                   Vector& norm,
                   int& p_dir);

    void createFile(string& filename, FILE*& fp);
        
    // general labels
    class FL_Labels {
    public:
      VarLabel* lastCompTimeLabel;
      VarLabel* fileVarsStructLabel;
      
      VarLabel* ICE_totalIntEngLabel;
      VarLabel* MPM_totalIntEngLabel;
      VarLabel* totalFluxesLabel;
    };
    
    FL_Labels* FL_lb;
    ICELabel* I_lb;
    MPMLabel* M_lb;
    
    enum FaceType {partial=0, entireFace=1, none=2};  
       
    struct cv_face{ 
      Point    startPt;
      Point    endPt;
      int      p_dir;
      Vector   normalDir;
      FaceType face;  
    };  
    
    std::map< int, cv_face* > d_cv_faces;
       
    //__________________________________
    // global constants
    SimulationStateP d_sharedState;
    Output* d_dataArchiver;
    ProblemSpecP d_prob_spec;
    
    MaterialSubset* d_zeroMatl;
    MaterialSet* d_zeroMatlSet;
    PatchSet* d_zeroPatch;
    
    double d_analysisFreq; 
    double d_StartTime;
    double d_StopTime;
    
  };
}

#endif
