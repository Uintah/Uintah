/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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


#ifndef MOMENTUM_ANALYSIS_H
#define MOMENTUM_ANALYSIS_H
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <map>

namespace Uintah {


/**************************************

CLASS
   momentumAnalysis

GENERAL INFORMATION

   momentumAnalysis.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


KEYWORDS
   momentumAnalysis

DESCRIPTION
   Performs a control volume analysis on the momentum field.

WARNING

****************************************/
  class momentumAnalysis : public AnalysisModule {
  public:
    momentumAnalysis(ProblemSpecP& prob_spec,
                     SimulationStateP& sharedState,
		       Output* dataArchiver);

    momentumAnalysis();

    virtual ~momentumAnalysis();

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

    void integrateMomentumField(const ProcessorGroup* pg,
                                const PatchSubset* patches,        
                                const MaterialSubset* matl_sub ,   
                                DataWarehouse* old_dw,             
                                DataWarehouse* new_dw);            

    void faceInfo(const std::string fc,
                   Patch::FaceType& face_side,
                   Vector& norm,
                   int& p_dir);

    void createFile(std::string& filename, FILE*& fp);

    void bulletProofing( GridP& grid,
                         const std::string& side,
                         const Point& start,
                         const Point& end );

    // VarLabels
    class MA_Labels {
      public:
        VarLabel* lastCompTime;
        VarLabel* fileVarsStruct;
        VarLabel* totalCVMomentum;
        VarLabel* CS_fluxes;
        VarLabel* vel_CC;
        VarLabel* rho_CC;
        VarLabel* uvel_FC;
        VarLabel* vvel_FC;
        VarLabel* wvel_FC;
        const VarLabel* delT;
    };

    MA_Labels* labels;

    enum FaceType {
      partialFace=0, 
      entireFace=1, 
      none=2 
    };

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

    int d_matlIndx;
    MaterialSet* d_matl_set;

  };
}

#endif
