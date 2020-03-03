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


#ifndef CONTROLVOLFLUXES_H
#define CONTROLVOLFLUXES_H
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/OnTheFlyAnalysis/controlVolume.h>
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
   controlVolFluxes

GENERAL INFORMATION

   fluxes.h

   Todd Harman
   Department of Mechanical Engineering
   University of Utah


DESCRIPTION
   computes the fluxes into/out of control volumes.

****************************************/
  class controlVolFluxes : public AnalysisModule {
  public:
    controlVolFluxes(const ProcessorGroup  * myworld,
                     const MaterialManagerP  materialManager,
                     const ProblemSpecP    & module_spec);

    controlVolFluxes();

    virtual ~controlVolFluxes();

    virtual void problemSetup(const ProblemSpecP & prob_spec,
                              const ProblemSpecP & restart_prob_spec,
                              GridP& grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc);

    virtual void outputProblemSpec(ProblemSpecP& ps){};

    virtual void scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleRestartInitialize(SchedulerP   & sched,
                                           const LevelP & level);

    virtual void restartInitialize(){};

    virtual void scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level);

    virtual void scheduleDoAnalysis_preReloc(SchedulerP   & sched,
                                             const LevelP & level) {};

  private:

    typedef controlVolume::FaceType cvFace;

    typedef std::map< cvFace, VarLabel*>   FaceLabelsMap;
    typedef std::map< cvFace, std::string> FaceNamesMap;


    void initialize(const ProcessorGroup *,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                          DataWarehouse  *,
                          DataWarehouse  * new_dw);

    void doAnalysis(const ProcessorGroup * pg,
                    const PatchSubset    * patches,
                    const MaterialSubset *,
                          DataWarehouse  *,
                          DataWarehouse  * new_dw);

    void integrate_Q_overCV(const ProcessorGroup  * pg,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matl_sub ,
                                  DataWarehouse   * old_dw,
                                  DataWarehouse   * new_dw);

    void createFile(std::string& filename,
                    FILE*& fp,
                    const controlVolume * cv);


    struct faceQuantities{
      std::map< int, double > Q_faceFluxes;      // flux across control volume faces
    };


    void initializeVars( faceQuantities* faceQ );

    //__________________________________
    //  accumulate fluxes across a control volume face
    template < class SFC_D >
    void integrate_Q_overFace( controlVolume::FaceType face,
                               const controlVolume * cv,
                               const Patch         * patch,
                               faceQuantities      * faceQ,
                               SFC_D               &  vel_FC,
                               constCCVariable<double>& rho_CC,
                               constCCVariable<Vector>& vel_CC);

    //__________________________________
    //  left flux - right flux
    Vector L_minus_R( std::map <int, Vector >& faceFlux);

    Vector L_minus_R( std::map <int, double >& faceFlux);

    //__________________________________
    //  For each CV create a vector of labels   desc_i
    std::vector<VarLabel*>
    createLabels( std::string desc,
                  const Uintah::TypeDescription* td );

    // VarLabels
    class labels {
      public:
        VarLabel* lastCompTime;
        VarLabel* fileVarsStruct;
        VarLabel* vel_CC;
        VarLabel* rho_CC;

        VarLabel* uvel_FC;
        VarLabel* vvel_FC;
        VarLabel* wvel_FC;

        // quanties of interest
        std::vector<std::string> Q_names;
        std::map<std::string, VarLabel*> Q_labels;

        // labels for each CV

        std::vector<VarLabel*>     totalQ_CV;
        std::vector<VarLabel*>     net_Q_faceFluxes;
        std::vector<FaceLabelsMap> Q_faceFluxes;
        std::vector<FaceNamesMap>  Q_faceNames;
    };

    labels* m_lb;

    std::vector<controlVolume*> m_controlVols;

    //__________________________________
    // global constants
    const MaterialSubset * m_zeroMatl;
    MaterialSet          * m_zeroMatlSet;
    PatchSet             * m_zeroPatch;
    const MaterialSubset * m_matl;
    MaterialSet          * m_matlSet;

    int m_col_width = 18;    //  column width used in output formatting
    int m_precision = 12;    // number of significant digits in output
    int m_matIdx;            // material index.


  };
}

#endif
