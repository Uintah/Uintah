/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef UINTAH_FSPICE_H
#define UINTAH_FSPICE_H

#include <CCA/Components/Application/ApplicationCommon.h>

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SwitchingCriteria.h>

#include <Core/Geometry/Vector.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>

namespace Uintah {

  class ICE;
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
  class SerialMPM;
  class SPICE;
  class SPICELabel;
  class FSPICELabel;

/*______________________________________________________________________
 FSPICE: SPICE coupled to MPM (Fluid-Structure SPICE)

 Schedules ICE, MPM and SPICE tasks the way MPMICE schedules ICE and MPM.
 It owns a SPICE object (d_spice, which owns the ICE object d_ice) and an
 MPM object (d_mpm), and carries its own copies of the MPMICE tasks: the
 MPMICE of the PIMPLE/KT development (private repo 8c10a44c8d) with the
 public MPM changes since (MPM reduction subsets, scheduleConservedQtys).
 MPMICE itself is not used and not modified.

 SPICE's tasks carry the gas-solid interface models (they are inert with no
 MPM materials):
   - KT next to solid cells (ktOpen face weights, solidCellFrac)
   - <interfaceTransport> donorAcceptor: no-penetration gas transport
   - <interfaceExchange>  particles | conduction (SPICE/Exchange)
   - <interfaceShear>     slip | model
 The tasks FSPICE adds for them:
   computeSolidVolFracP     sharp solid fraction from the particle volume
 and the frozen-solid PITCH outer loop: MPM runs once per step (the k=0
 pass); the outer iterations re-run the gas tasks only
 (scheduleGasChainForPitch, handed to SPICE through its PITCH hooks),
 with the solid inputs copied from k=0.

 MPMICE tasks FSPICE changed (copies of MPMICE):
   computeEquilibrationPressure  placeholder trace gas, gas-only pressure
                                 from its mass, trace-gas compressibility
   problemSetup                  rigid-contact warning, PITCH hooks

 Input: <SimulationComponent type="fspice"/> (SerialMPM) or "rfspice"
 (RigidMPM), the <CFD><SPICE> block and an <MPM> block.  With KT on and MPM
 materials the interface options default to donorAcceptor + conduction +
 model.  See SPICE/Docs/PIMPLE_KT_method.pdf.

 Limitations: single level (no AMR), exactly one ICE material.  With no MPM
 materials FSPICE runs as SPICE.  PITCH with MPM materials needs the scalar
 exchange model without convective heat transfer.  <interfaceShear>
 slip/model is for grid-aligned walls.
______________________________________________________________________*/

  class FSPICE : public ApplicationCommon {

  public:
    enum SolidType { STANDARD_MPM = 0, RIGID_MPM };

    FSPICE( const ProcessorGroup * myworld,
            const MaterialManagerP materialManager,
            const SolidType        solidType );

    virtual ~FSPICE();

    virtual double recomputeDelT( const double delT );

    virtual void problemSetup( const ProblemSpecP & params,
                               const ProblemSpecP & restart_prob_spec,
                               GridP              & grid );

    virtual void outputProblemSpec( ProblemSpecP & ps );

    virtual void scheduleInitialize( const LevelP & level,
                                     SchedulerP   & );

    virtual void scheduleRestartInitialize( const LevelP & level,
                                            SchedulerP   & sched );

    virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                                SchedulerP   & );

    virtual void scheduleTimeAdvance( const LevelP & level,
                                      SchedulerP   & );

    virtual void scheduleFinalizeTimestep( const LevelP & level,
                                           SchedulerP   & );

    virtual void scheduleAnalysis( const LevelP & level,
                                   SchedulerP   & sched );

  protected:

    void scheduleInterpolateNCToCC_0( SchedulerP&,
                                      const PatchSet*,
                                      const MaterialSubset*,
                                      const MaterialSet* );

    void scheduleComputeGridVelocityStarBC( SchedulerP&,
                                            const PatchSet*,
                                            const MaterialSet* );

    void scheduleComputeLagrangianValuesMPM( SchedulerP&,
                                             const PatchSet*,
                                             const MaterialSubset*,
                                             const MaterialSet* );

    void scheduleInterpolateCCToNC( SchedulerP&,
                                    const PatchSet*,
                                    const MaterialSet* );

    void scheduleComputeCCVelAndTempRates( SchedulerP&,
                                           const PatchSet*,
                                           const MaterialSet* );

    void scheduleComputePressure( SchedulerP&,
                                  const PatchSet*,
                                  const MaterialSubset*,
                                  const MaterialSubset*,
                                  const MaterialSubset*,
                                  const MaterialSet* );

    void scheduleInterpolatePressCCToPressNC( SchedulerP&,
                                              const PatchSet*,
                                              const MaterialSubset*,
                                              const MaterialSet* );

    void scheduleInterpolatePAndGradP( SchedulerP&,
                                       const PatchSet*,
                                       const MaterialSubset*,
                                       const MaterialSubset*,
                                       const MaterialSubset*,
                                       const MaterialSet* );

    void scheduleParticleRelocation( SchedulerP        & sched,
                                     const LevelP      & level,
                                     const MaterialSet * mpm_matls );

    //__________________________________
    //  PITCH outer loop: gas-only sub-graph with the MPM solid frozen at k=0
    void schedulePitchCaptureSolid0( SchedulerP&,
                                      const PatchSet*,
                                      const MaterialSet* mpm_matls );

    void pitchCaptureSolid0( const ProcessorGroup*,
                              const PatchSubset*,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

    void schedulePitchCopyFrozenSolid( SchedulerP&,
                                        const PatchSet*,
                                        const MaterialSet* mpm_matls );

    void pitchCopyFrozenSolid( const ProcessorGroup*,
                                const PatchSubset*,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw );

    void scheduleGasChainForPitch( SchedulerP&,
                                    const LevelP& );

    void pitchFrozenSolidRequires( Task* t );

    //__________________________________
    //  <interfaceTransport> donorAcceptor: each cell's solid volume fraction
    //  from the particles' volume on the nodes (no node smearing)
    void scheduleComputeSolidVolFracP( SchedulerP&,
                                       const PatchSet*,
                                       const MaterialSubset* ice_matls,
                                       const MaterialSubset* mpm_matls,
                                       const MaterialSet* all_matls );

    void computeSolidVolFracP( const ProcessorGroup*,
                               const PatchSubset*,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw );

    void schedulePitchCopySolidVolFracP( SchedulerP&,
                                          const PatchSet*,
                                          const MaterialSet* ice_matls );

    void pitchCopySolidVolFracP( const ProcessorGroup*,
                                  const PatchSubset*,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw );

    //______________________________________________________________________
    //       A C T U A L   S T E P S :
    void actuallyInitialize( const ProcessorGroup*,
                             const PatchSubset* patch,
                             const MaterialSubset* matls,
                             DataWarehouse*,
                             DataWarehouse* new_dw );

    void interpolateNCToCC_0( const ProcessorGroup*,
                              const PatchSubset* patch,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

    void computeGridVelocityStarBC( const ProcessorGroup*,
                                    const PatchSubset* patch,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw );

    void computeLagrangianValuesMPM( const ProcessorGroup*,
                                     const PatchSubset* patch,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw );

    void computeEquilibrationPressure( const ProcessorGroup*,
                                       const PatchSubset* patch,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse*,
                                       const MaterialSubset* press_matl );

    void interpolateCCToNC( const ProcessorGroup*,
                            const PatchSubset* patch,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

    void computeCCVelAndTempRates( const ProcessorGroup*,
                                   const PatchSubset* patch,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw );

    void interpolatePressCCToPressNC( const ProcessorGroup*,
                                      const PatchSubset* patch,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw );

    void interpolatePAndGradP( const ProcessorGroup*,
                               const PatchSubset* patch,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw );

    void binaryPressureSearch( std::vector<constCCVariable<double> >& Temp,
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

    virtual void scheduleSwitchTest( const LevelP& level, SchedulerP& sched );

  private:
    FSPICE( const FSPICE& );
    FSPICE& operator=( const FSPICE& );

    MPMLabel*        Mlb{nullptr};
    ICELabel*        Ilb{nullptr};
    MPMICELabel*     MIlb{nullptr};
    SPICELabel*      Slb{nullptr};          // owned by d_spice
    FSPICELabel*     FSlb{nullptr};         // labels only FSPICE computes

    bool             d_rigidMPM{false};
    SerialMPM*       d_mpm{nullptr};
    SPICE*           d_spice{nullptr};      // the CFD side: SPICE, which owns ICE
    ICE*             d_ice{nullptr};        // d_spice's ICE object
    int              d_8or27{8};
    int              NGN{1};
    bool             d_testForNegTemps_mpm{true};

    //  No MPM materials in the input: FSPICE forwards every call to d_spice
    //  and runs exactly as SPICE.
    bool             d_gasOnly{false};

    std::vector<AnalysisModule*>  d_analysisModules;

    SwitchingCriteria* d_switchCriteria{nullptr};

    double d_SMALL_NUM;
    double d_TINY_RHO;
  };

} // End namespace Uintah

#endif
