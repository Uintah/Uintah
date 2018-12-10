/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_ontheflyAnalysis_particleExtract_h
#define Packages_Uintah_CCA_Components_ontheflyAnalysis_particleExtract_h
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>

#include <map>
#include <vector>

namespace Uintah {
  

/**************************************

CLASS
   particleExtract
   
GENERAL INFORMATION

   particleExtract.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   particleExtract

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class particleExtract : public AnalysisModule {
  public:
    particleExtract(const ProcessorGroup* myworld,
                    const MaterialManagerP materialManager,
                    const ProblemSpecP& module_spec);
    
    particleExtract();
                    
    virtual ~particleExtract();
   
    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};
    
    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level){};
                                    
    virtual void restartInitialize();
                                    
    virtual void scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level);
    
    virtual void scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                    const LevelP& level);
   
  private:

    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
                    
    void doAnalysis_preReloc(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
    
    void doAnalysis(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
                    
    void createFile(std::string& filename, FILE*& fp);
    
    void createDirectory(std::string& lineName, std::string& levelIndex);
    
    bool doMPMOnLevel(int level, int numLevels);
                    
    
    // general labels
    class particleExtractLabel {
    public:
      VarLabel* lastWriteTimeLabel;
      VarLabel* filePointerLabel;
      VarLabel* filePointerLabel_preReloc;
    };
    
    particleExtractLabel* ps_lb;
    MPMLabel* M_lb;
       
    //__________________________________
    // global constants
    double d_writeFreq; 
    double d_startTime;
    double d_stopTime;
    double d_colorThreshold;
    std::vector<VarLabel*> d_varLabels;

    const Material* d_matl;
    MaterialSet* d_matl_set;
    MaterialSubset* d_matl_subset;
    std::set<std::string> d_isDirCreated;        
  };
}

#endif
