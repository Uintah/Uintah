/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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


#ifndef CCA_Components_ontheflyAnalysis_SGS_ReynoldsStress_h
#define CCA_Components_ontheflyAnalysis_SGS_ReynoldsStress_h

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <vector>

namespace Uintah {

  class ICELabel;

/**************************************

CLASS
   SGS_ReynoldsStress

GENERAL INFORMATION

   SGS_ReynoldsStress.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah

KEYWORDS
   SGS_ReynoldsStress

DESCRIPTION
   This module computes the sub grid scale Reynolds Stresses at the cell center

WARNING

****************************************/
  class SGS_ReynoldsStress : public AnalysisModule {
  public:
    SGS_ReynoldsStress(const ProcessorGroup   * myworld,
                       const MaterialManagerP materialManager,
                       const ProblemSpecP     & module_spec);

    SGS_ReynoldsStress();

    virtual ~SGS_ReynoldsStress();

    virtual void problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP  & sched,
                                    const LevelP& level){};

    virtual void scheduleRestartInitialize(SchedulerP   & sched,
                                           const LevelP & level){};

    virtual void scheduleDoAnalysis(SchedulerP  & sched,
                                    const LevelP& level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

  private:

    void doAnalysis(const ProcessorGroup  * pg,
                    const PatchSubset     * patches,
                    const MaterialSubset  *,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);


    void interpolateTauComponents( const Patch* patch,
                                   constSFCXVariable<Vector>& tau_X_FC,
                                   constSFCYVariable<Vector>& tau_Y_FC,
                                   constSFCZVariable<Vector>& tau_Z_FC,
                                   CCVariable<Matrix3>      & SGS_ReynoldsStress );

    void interpolateTauX_driver( CellIterator iterLimits,
                                 const Vector dx,
                                 constSFCXVariable<Vector>& tau_X_FC,
                                 CCVariable<Matrix3>      & SGS_ReynoldsStress);

    void interpolateTauY_driver( CellIterator iterLimits,
                                 const Vector dx,
                                 constSFCYVariable<Vector>& tau_Y_FC,
                                 CCVariable<Matrix3>      & SGS_ReynoldsStress);

    void interpolateTauZ_driver( CellIterator iterLimits,
                                 const Vector dx,
                                 constSFCZVariable<Vector>& tau_Z_FC,
                                 CCVariable<Matrix3>      & SGS_ReynoldsStress );


    // general labels
    VarLabel* SGS_ReynoldsStressLabel;

    ICELabel* I_lb;

    //__________________________________
    // global constants
    const Material      * m_matl      {nullptr};
    MaterialSet         * m_matl_set  {nullptr};
    const MaterialSubset* m_matl_sub  {nullptr};

  };
}

#endif
