/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_HOMEBREW_MPMICE_H
#define UINTAH_HOMEBREW_MPMICE_H

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {
  class ICE;
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
  class Output;

/**************************************

CLASS
   MPMICE

   Short description...

GENERAL INFORMATION

   MPMICE.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   MPMICE

DESCRIPTION
   Long description...

WARNING

****************************************/

enum MPMType {STAND_MPMICE = 0, RIGID_MPMICE, SHELL_MPMICE, FRACTURE_MPMICE};

class MPMICE : public ApplicationCommon {

public:
  MPMICE(const ProcessorGroup* myworld,
         const MaterialManagerP materialManager,
         MPMType type, const bool doAMR = false);

  virtual ~MPMICE();

  virtual double recomputeDelT(const double delT);

  virtual void problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& restart_prob_spec,
                            GridP& grid);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  virtual void scheduleRestartInitialize(const LevelP& level,
                                         SchedulerP& sched);

  virtual void scheduleComputeStableTimeStep(const LevelP& level,
                                             SchedulerP&);

  // scheduleTimeAdvance version called by the AMR simulation controller.
  virtual void scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP&);

  virtual void scheduleFinalizeTimestep(const LevelP& level,
                                        SchedulerP&);

protected:
  void scheduleInterpolateNCToCC_0(SchedulerP&,
                                  const PatchSet*,
                                  const MaterialSubset*,
                                  const MaterialSet*);

  void scheduleCoarsenCC_0(SchedulerP&,
                           const PatchSet*,
                           const MaterialSet*);

  void scheduleCoarsenNCMass(SchedulerP&,
                             const PatchSet*,
                             const MaterialSet*);

  void scheduleComputeLagrangianValuesMPM(SchedulerP&,
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);

  void scheduleCoarsenLagrangianValuesMPM(SchedulerP&,
                                          const PatchSet*,
                                          const MaterialSet*);

  void scheduleInterpolateCCToNC(SchedulerP&,
                                 const PatchSet*,
                                 const MaterialSet*);

  void scheduleComputeCCVelAndTempRates(SchedulerP&,
                                        const PatchSet*,
                                        const MaterialSet*);

  void scheduleRefineCC(SchedulerP&,
                        const PatchSet*,
                        const MaterialSet*);

  void scheduleComputePressure(SchedulerP&,
                               const PatchSet*,
                               const MaterialSubset*,
                               const MaterialSubset*,
                               const MaterialSubset*,
                               const MaterialSet*);


  void scheduleInterpolatePressCCToPressNC(SchedulerP&,
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);

  void scheduleRefinePressCC(SchedulerP&,
                             const PatchSet*,               
                             const MaterialSubset*,         
                             const MaterialSet*);           

  void scheduleInterpolatePAndGradP(SchedulerP&,
                                    const PatchSet*,
                                    const MaterialSubset*,
                                    const MaterialSubset*,
                                    const MaterialSubset*,
                                    const MaterialSet*);

  void scheduleParticleRelocation( SchedulerP        & sched,
                                   const LevelP      & level,
                                   const MaterialSet * mpm_matls);

//______________________________________________________________________
//       A C T U A L   S T E P S :
  void actuallyInitialize(const ProcessorGroup*,
                          const PatchSubset* patch,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw);


  void interpolateNCToCC_0(const ProcessorGroup*,
                           const PatchSubset* patch,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);

  void computeLagrangianValuesMPM(const ProcessorGroup*,
                                  const PatchSubset* patch,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  void computeEquilibrationPressure(const ProcessorGroup*,
                                    const PatchSubset* patch,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse*,
                                    const MaterialSubset* press_matl);


  void interpolateCCToNC(const ProcessorGroup*,
                         const PatchSubset* patch,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw);

  void computeCCVelAndTempRates(const ProcessorGroup*,
                                const PatchSubset* patch,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

  void interpolatePressCCToPressNC(const ProcessorGroup*,
                                   const PatchSubset* patch,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  void interpolatePAndGradP(const ProcessorGroup*,
                            const PatchSubset* patch,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);


  void binaryPressureSearch(std::vector<constCCVariable<double> >& Temp,
                            std::vector<CCVariable<double> >& rho_micro,
                            std::vector<CCVariable<double> >& vol_frac,
                            std::vector<CCVariable<double> >& rho_CC_new,
                            std::vector<CCVariable<double> >& speedSound_new,
                            std::vector<double> & dp_drho,
                            std::vector<double> & dp_de,
                            std::vector<double> & press_eos,
                            constCCVariable<double> & press,
                            CCVariable<double> & press_new,
                            double press_ref,
                            std::vector<constCCVariable<double> > & cv,
                            std::vector<constCCVariable<double> > & gamma,
                            double convergence_crit,
                            unsigned int numALLMatls,
                            int & count,
                            double & sum,
                            IntVector c );

  virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

  //__________________________________
  //   AMR
  virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                       SchedulerP& scheduler,
                                       bool needOld, bool needNew);

  virtual void scheduleRefine (const PatchSet* patches,
                               SchedulerP& sched);

  virtual void scheduleCoarsen(const LevelP& coarseLevel,
                               SchedulerP& sched);

  void refine(const ProcessorGroup*,
              const PatchSubset* patches,
              const MaterialSubset* matls,
              DataWarehouse*,
              DataWarehouse* new_dw);

  virtual void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                            SchedulerP& sched);

  virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                     SchedulerP& sched);

  void scheduleRefineVariableCC(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const VarLabel* variable);

  template<typename T>
    void scheduleCoarsenVariableCC(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const VarLabel* variable,
                                   T defaultValue,
                                   bool modifies,
                                   const std::string& coarsenMethod);

  template<typename T>
    void scheduleCoarsenVariableNC(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const VarLabel* variable,
                                   T defaultValue,
                                   bool modifies,
                                   std::string coarsenMethod);

  template<typename T>
    void refineVariableCC(const ProcessorGroup*,
                          const PatchSubset* patch,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const VarLabel* variable);

//
  template<typename T>
    void coarsenDriver_stdNC(IntVector cl,
                             IntVector ch,
                             IntVector refinementRatio,
                             double ratio,
                             const Level* coarseLevel,
                             constNCVariable<T>& fine_q_NC,
                             NCVariable<T>& coarse_q_NC );



  template<typename T>
    void coarsenVariableCC(const ProcessorGroup*,
                           const PatchSubset* patch,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const VarLabel* variable,
                           T defaultValue,
                           bool modifies,
                           std::string coarsenMethod);

  template<typename T>
    void coarsenVariableNC(const ProcessorGroup*,
                           const PatchSubset* patch,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const VarLabel* variable,
                           T defaultValue,
                           bool modifies,
                           std::string coarsenMethod);

    void refineCoarseFineInterface(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* fine_old_dw,
                                   DataWarehouse* fine_new_dw);

protected:
  MPMICE(const MPMICE&);
  MPMICE& operator=(const MPMICE&);

  MPMLabel*        Mlb;
  ICELabel*        Ilb;
  MPMICELabel*     MIlb;

  bool             d_rigidMPM;
  SerialMPM*       d_mpm;
  ICE*             d_ice;
  int              d_8or27;
  int              NGN;
  bool             d_testForNegTemps_mpm;
  bool             do_mlmpmice;

  std::vector<AnalysisModule*>  d_analysisModules;

  SwitchingCriteria* d_switchCriteria;

  double d_SMALL_NUM;
  double d_TINY_RHO;
};

} // End namespace Uintah

#endif
