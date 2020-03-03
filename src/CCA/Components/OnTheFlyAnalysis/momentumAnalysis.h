/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
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
    momentumAnalysis(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager,
                     const ProblemSpecP& module_spec);

    momentumAnalysis();

    virtual ~momentumAnalysis();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP& sched,
                                    const LevelP& level);
                                    
    virtual void scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level);

    virtual void restartInitialize(){};

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

    void faceInfo( const std::string fc,
                   Vector& norm,
                   int& p_dir,
                   int& index);

    void createFile(std::string& filename, FILE*& fp);

    void bulletProofing( GridP& grid,
                         const std::string& side,
                         const Point& start,
                         const Point& end );

    VarLabel* assignLabel( const std::string& desc );



    struct faceQuantities{
      std::map< int, Vector > convect_faceFlux;      // convective momentum flux across control volume faces
      std::map< int, Vector > viscous_faceFlux;      // viscous  momentum flux across control volume faces
      std::map< int, double > pressForce_face;       // pressure force on each face
    };

    void initializeVars( faceQuantities* faceQ );

    //__________________________________
    //  accumulate fluxes across a control volume face
    template < class SFC_D, class SFC_V >
    void integrateOverFace( const std::string faceName,
                            const double faceArea,
                            CellIterator iterLimits,
                            faceQuantities* faceQ,
                            SFC_D& vel_FC,
                            SFC_D& press_FC,
                            SFC_V& tau_FC,
                            constCCVariable<double>& rho_CC,
                            constCCVariable<Vector>& vel_CC );

    //__________________________________
    //  left flux - right flux
    Vector L_minus_R( std::map <int, Vector >& faceFlux);

    // VarLabels
    class MA_Labels {
      public:
        VarLabel* lastCompTime;
        VarLabel* fileVarsStruct;

        VarLabel* totalCVMomentum;
        VarLabel* convectMom_fluxes;
        VarLabel* viscousMom_fluxes;
        VarLabel* pressForces;

        VarLabel* vel_CC;
        VarLabel* rho_CC;

        VarLabel* uvel_FC;
        VarLabel* vvel_FC;
        VarLabel* wvel_FC;

        VarLabel* pressX_FC;
        VarLabel* pressY_FC;
        VarLabel* pressZ_FC;

        VarLabel* tau_X_FC;
        VarLabel* tau_Y_FC;
        VarLabel* tau_Z_FC;
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
    MaterialSubset * d_zeroMatl;
    MaterialSubset * d_pressMatl;
    MaterialSet    * d_zeroMatlSet;
    PatchSet       * d_zeroPatch;

    int d_matlIndx;                      // material index.
    int d_pressIndx;                     // pressure matl index
    MaterialSet* d_matl_set;

  };
}

#endif
