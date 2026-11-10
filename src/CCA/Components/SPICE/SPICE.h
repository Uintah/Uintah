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


#ifndef UINTAH_SPICE_H
#define UINTAH_SPICE_H

// Advector.h must be included before ApplicationCommon.h (see ICE.h).
#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Components/Application/ApplicationCommon.h>

#include <CCA/Components/SPICE/Core/CentralUpwind.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <functional>
#include <string>
#include <vector>

namespace Uintah {

  class Advector;
  class ICE;
  class ICELabel;
  class SPICELabel;
  class VarLabel;

/*______________________________________________________________________
 SPICE: Semi-implicit Pressure Iterated Compressible EOS

 A component built on ICE the way MPMICE is: it owns an ICE object
 (d_ice), lets ICE parse the input and schedule every task SPICE does not
 change, and implements the new and modified tasks itself:

   - the PITCH outer loop (Picard ITeration of the fluid CHain): a
     sub-scheduler that re-runs the fluid chain with a Crank-Nicolson
     theta blend of the conserved state
   - Kurganov-Tadmor (KT, "CentralUpwind") face fluxes, face velocity,
     face pressure and advection
   - the consistent pressure-work formulation (always on in SPICE)

 Input: <CFD><SPICE> holds every stock ICE CFD option plus <CentralUpwind>
 and <PITCH>.  See StandAlone/inputs/UPS_SPEC/spice_spec.xml and
 Docs/PIMPLE_KT_method.pdf.

 Tasks SPICE owns, copied from ICE and then modified.  ICE marks each
 original with "SPICE owns a modified copy"; port ICE fixes to both.
   accumulateEnergySourceSinks      always (new pressure work)
   accumulateMomentumSourceSinks    always (<interfaceShear> model)
   computeLagrangianSpecificVolume  always (thermal expansion scaled by TMV)
   advectAndAdvanceInTime           always (total-energy advection, KT fluxes)
   computeVel_FC                    KT with <faceVelocity> on
   computePressFC                   KT with <facePressure> on
   computeDelPressAndUpdatePressCC  KT on
 Every other task is ICE's, scheduled through d_ice.

 MPM materials: the tasks above also carry the gas-solid interface models
 FSPICE (SPICE + MPM) uses: KT next to solid cells (ktOpen face weights),
 no-penetration transport (<interfaceTransport> donorAcceptor), the
 particle-aware exchange (SPICE/Exchange), interface conduction and wall
 shear, and the frozen-solid PITCH restart.  All of it is inert without
 MPM materials.  The stand-alone SPICE component rejects MPM materials;
 FSPICE runs SPICE "hosted" (setHostedByFSPICE) on its own ICE setup.
 See Docs/PIMPLE_KT_method.pdf.

 Limitations: single level (no AMR), exactly one ICE material (a
 single-fluid code; species are transported scalars, e.g. gasCombustion).
 PITCH does not support impICE or a turbulence model; KT does not
 support impICE.
______________________________________________________________________*/

  class SPICE : public ApplicationCommon {

  public:
    SPICE( const ProcessorGroup  * myworld,
           const MaterialManagerP  materialManager,
           const bool              doAMR = false );

    virtual ~SPICE();

    virtual double recomputeDelT( const double delT );

    virtual void problemSetup( const ProblemSpecP & params,
                               const ProblemSpecP & restart_prob_spec,
                               GridP              & grid );

    virtual void outputProblemSpec( ProblemSpecP & ps );

    virtual void scheduleInitialize( const LevelP & level,
                                     SchedulerP   & sched );

    virtual void scheduleRestartInitialize( const LevelP & level,
                                            SchedulerP   & sched );

    virtual void scheduleComputeStableTimeStep( const LevelP & level,
                                                SchedulerP   & sched );

    virtual void scheduleTimeAdvance( const LevelP & level,
                                      SchedulerP   & sched );

    virtual void scheduleFinalizeTimestep( const LevelP & level,
                                           SchedulerP   & sched );

    virtual void scheduleAnalysis( const LevelP & level,
                                   SchedulerP   & sched );

    //__________________________________
    //  The fluid-dynamics portion of one timestep, from the thermo/transport
    //  properties through advection.  Called once for the main graph and,
    //  with inOuterLoop = true, to build the PITCH sub-graph.
    void scheduleFluidChain( SchedulerP   & sched,
                             const LevelP & level,
                             const bool     inOuterLoop );

    //__________________________________
    //  Schedule functions for the tasks SPICE owns
    void scheduleComputeVel_FC( SchedulerP          &,
                                const PatchSet      *,
                                const MaterialSubset*,
                                const MaterialSubset*,
                                const MaterialSubset*,
                                const MaterialSet   * );

    void scheduleComputeCentralWeights( SchedulerP        &,
                                        const PatchSet    *,
                                        const MaterialSet * );

    void scheduleComputeCentralFluxes( SchedulerP        &,
                                       const PatchSet    *,
                                       const MaterialSet * );

    void scheduleComputeDelPressAndUpdatePressCC( SchedulerP          &,
                                                  const PatchSet      *,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSet   * );

    void scheduleComputePressFC( SchedulerP          &,
                                 const PatchSet      *,
                                 const MaterialSubset*,
                                 const MaterialSet   * );

    void scheduleAccumulateMomentumSourceSinks( SchedulerP          &,
                                                const PatchSet      *,
                                                const MaterialSubset*,
                                                const MaterialSubset*,
                                                const MaterialSubset*,
                                                const MaterialSet   * );

    void scheduleComputeLagrangianSpecificVolume( SchedulerP          &,
                                                  const PatchSet      *,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSubset*,
                                                  const MaterialSet   * );

    void scheduleAccumulateEnergySourceSinks( SchedulerP          &,
                                              const PatchSet      *,
                                              const MaterialSubset*,
                                              const MaterialSubset*,
                                              const MaterialSubset*,
                                              const MaterialSet   * );

    void scheduleAdvectAndAdvanceInTime( SchedulerP          &,
                                         const PatchSet      *,
                                         const MaterialSubset*,
                                         const MaterialSet   * );

    //__________________________________
    //  P I M P L E   outer loop (sub-scheduler)
    void schedulePitchCopyModelSources( SchedulerP        & sched,
                                         const LevelP      & level,
                                         const MaterialSet * matls );

    void scheduleCapturePitchAdv0( SchedulerP        & sched,
                                    const LevelP      & level,
                                    const MaterialSet * matls );

    void schedulePitchBlend( SchedulerP        & sched,
                              const LevelP      & level,
                              const MaterialSet * ice_matls );

    void schedulePitchResidual( SchedulerP        & sched,
                                 const LevelP      & level,
                                 const MaterialSet * ice_matls );

    void schedulePitchOuterLoop( SchedulerP   & sched,
                                  const LevelP & level );

    //__________________________________
    //  Used by FSPICE (SPICE + MPM).  Both must be called before problemSetup.
    //  setHostedByFSPICE: allow MPM materials.  The PITCH hooks replace the
    //  sub-graph's fluid chain with FSPICE's gas-only chain (MPM solid frozen
    //  at k=0) and add the frozen-solid requires to the outer-loop task.
    void setHostedByFSPICE() {
      d_hostedByFSPICE = true;
    };

    void setPitchHooks( const std::function<void(SchedulerP&, const LevelP&)> & chain,
                         const std::function<void(Task*)>                       & requiresHook )
    {
      d_pitchChainHook    = chain;
      d_pitchRequiresHook = requiresHook;
    };

  protected:

    //__________________________________
    //  Task callbacks
    void computeCentralWeights( const ProcessorGroup *,
                                const PatchSubset    *,
                                const MaterialSubset *,
                                DataWarehouse        *,
                                DataWarehouse        * );

    void computeCentralFluxes( const ProcessorGroup *,
                               const PatchSubset    *,
                               const MaterialSubset *,
                               DataWarehouse        *,
                               DataWarehouse        * );

    void computeVel_FC( const ProcessorGroup *,
                        const PatchSubset    *,
                        const MaterialSubset *,
                        DataWarehouse        *,
                        DataWarehouse        * );

    void computeDelPressAndUpdatePressCC( const ProcessorGroup *,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * matls,
                                          DataWarehouse        *,
                                          DataWarehouse        * );

    void computePressFC( const ProcessorGroup *,
                         const PatchSubset    * patches,
                         const MaterialSubset * matls,
                         DataWarehouse        *,
                         DataWarehouse        * );

    void accumulateMomentumSourceSinks( const ProcessorGroup *,
                                        const PatchSubset    * patches,
                                        const MaterialSubset * matls,
                                        DataWarehouse        *,
                                        DataWarehouse        * );

    void computeLagrangianSpecificVolume( const ProcessorGroup *,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * matls,
                                          DataWarehouse        *,
                                          DataWarehouse        * );

    //  <interfaceExchange> conduction (MPM materials): heat conducted across
    //  the gas/solid interface, as energy sources (J per step) for the gas
    //  and for each MPM material (in MPM material order)
    void interfaceConduction( const Patch                       * patch,
                              DataWarehouse                     * old_dw,
                              DataWarehouse                     * new_dw,
                              const double                        delT,
                              CCVariable<double>                & gasSrc,
                              std::vector< CCVariable<double> > & solidSrc );

    //  <interfaceShear> model (MPM materials): wall shear between the gas and
    //  each MPM material (momentum per step) and the heat it dissipates
    void interfaceShear( const Patch                       * patch,
                         DataWarehouse                     * old_dw,
                         DataWarehouse                     * new_dw,
                         const double                        delT,
                         CCVariable<Vector>                & gasMom,
                         std::vector< CCVariable<Vector> > & solidMom,
                         CCVariable<double>                * gasHeat );

    void accumulateEnergySourceSinks( const ProcessorGroup *,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                      DataWarehouse        *,
                                      DataWarehouse        * );

    void advectAndAdvanceInTime( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * matls,
                                 DataWarehouse        *,
                                 DataWarehouse        * );

    void pitchCopyModelSources( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * matls,
                                 DataWarehouse        * old_dw,
                                 DataWarehouse        * new_dw );

    void capturePitchAdv0( const ProcessorGroup *,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw );

    void pitchBlend( const ProcessorGroup *,
                      const PatchSubset    * patches,
                      const MaterialSubset * matls,
                      DataWarehouse        * old_dw,
                      DataWarehouse        * new_dw );

    void pitchResidual( const ProcessorGroup *,
                         const PatchSubset    * patches,
                         const MaterialSubset * matls,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw );

    //  MPM materials: reset the gas in solid cells of the sub OldDW to time n
    //  before each outer iteration (the solid is frozen at n)
    void pitchRestartSolidCells( const PatchSubset    * patches,
                                  const MaterialSubset * ice_mss,
                                  DataWarehouse        * pOldDW,
                                  DataWarehouse        * pNewDW,
                                  DataWarehouse        * subOldDW );

    void pitchOuterLoop( const ProcessorGroup * pg,
                          const PatchSubset    * patches,
                          const MaterialSubset * matls,
                          DataWarehouse        * ParentOldDW,
                          DataWarehouse        * ParentNewDW,
                          LevelP                 level );

    //__________________________________
    //  KT face-reconstruction info passed into computeVelFace /
    //  computePressFace.  alphaOwn is the per-face own/nei blend weight
    //  (ktAlphaOwn_FC{X,Y,Z}) for this face direction; lo/hi/periodic are
    //  the level's cell index range and periodicity, needed by
    //  centralUpwind::ktStencilFlags() at domain boundaries.  AT is the
    //  const face-variable type for this direction.
    template<class AT>
    struct KTFaceVelInfo {
      const AT * alphaOwn{nullptr};
      IntVector  lo;
      IntVector  hi;
      IntVector  periodic;
      const AT * open{nullptr};   // ktOpen_FC (MPM materials only), else nullptr
    };

    template<class AT>
    struct KTFacePressInfo {
      const AT * alphaOwn{nullptr};
      IntVector  lo;
      IntVector  hi;
      IntVector  periodic;
      const AT * open{nullptr};   // ktOpen_FC (MPM materials only), else nullptr
    };

    template<class T, class AT> void computeVelFace( int dir, CellIterator it,
                                                     IntVector adj_offset,
                                                     double dx,
                                                     double delT,
                                                     double gravity,
                                                     constCCVariable<double>& rho_CC,
                                                     constCCVariable<double>& sp_vol_CC,
                                                     constCCVariable<Vector>& vel_CC,
                                                     constCCVariable<double>& press_CC,
                                                     T& vel_FC,
                                                     T& gradP_FC,
                                                     bool include_acc,
                                                     const KTFaceVelInfo<AT>* kt = nullptr );

    template<class T, class AT> void computePressFace( CellIterator it,
                                                       int dir,
                                                       IntVector adj_offset,
                                                       constCCVariable<double>& sum_rho,
                                                       constCCVariable<double>& press_CC,
                                                       T& press_FC,
                                                       const KTFacePressInfo<AT>* kt = nullptr );

    //__________________________________
    //  Advection operator factory handed to ICE (ICE::setAdvectorFactory):
    //  builds a CentralUpwindAdvector for <advection type="CentralUpwind">
    //  and defers every other type to ICE's AdvectionFactory.
    Advector* createAdvector( ProblemSpecP & cfd_ps,
                              bool         & useCompatibleFluxes,
                              int          & orderOfAdvection );

    //  The per-transported-variable "*_adv0" labels, indexed in the nested
    //  (model, transported variable) order every loop over d_ice->d_models
    //  uses.
    const VarLabel* pitchTvarAdv0Label( int & idx ) const
    {
      return d_pitchTvarAdv0Labels.at(idx++);
    }

    //__________________________________
    ICE        * d_ice{nullptr};    // the ICE component SPICE builds on
    ICELabel   * lb{nullptr};       // ICE's labels (owned by d_ice), named as in ICE
                                    // so the copied tasks read like their originals
    SPICELabel * Slb{nullptr};      // labels only SPICE computes

    Ghost::GhostType  m_gn  = Ghost::None;
    Ghost::GhostType  m_gac = Ghost::AroundCells;

    //__________________________________
    //  CentralUpwind (Kurganov-Tadmor) options, <CFD><SPICE><CentralUpwind>
    bool        d_useCentralUpwind{false};              // <advection type="CentralUpwind">
    std::string d_ktLimiterName{"vanAlbada"};
    std::string d_ktFluxScheme{"Kurganov"};              // Kurganov | Tadmor
    std::string d_ktKappaFunction{"MachToAcCourant"};    // MachToAcCourant | one
    bool        d_ktFaceVel{true};
    bool        d_ktFacePress{true};
    bool        d_ktPressureFlux{true};

    //  KT next to MPM solids, <CFD><SPICE><CentralUpwind>.  Read only when
    //  MPM materials exist; the defaults then are donorAcceptor, conduction
    //  and model (see problemSetup).
    double d_ktSolidCellFrac{1.0e-3};   // <solidCellFrac>: MPM solid vol-fraction above which a cell counts as solid (KT off on its faces)
    bool   d_ktInterfaceDA{false};      // <interfaceTransport> donorAcceptor: no-penetration gas transport at gas/solid faces
    double d_ktCoveredSolid{0.9};       // solid fraction (solidVolFracP_CC) at or above which a cell counts as covered by the solid
    bool   d_ktInterfaceExch{false};    // <interfaceExchange> particles: no gas-MPM heat exchange where no particles are
    bool   d_ktInterfaceCond{false};    // <interfaceExchange> conduction: particles + conduction across the gas/solid interface
    bool   d_ktInterfaceSlip{false};    // <interfaceShear> slip: no tangential gas-MPM momentum exchange where no particles are
    bool   d_ktInterfaceShear{false};   // <interfaceShear> model: slip + the wall shear of SPICE::interfaceShear

    //  KT with MPM materials: the MPM materials' vol_frac in the pressure
    //  equation keeps a SecondOrder advector (the KT one is for the gas only)
    Advector * d_mpmAdvector{nullptr};

    centralUpwind::Limiter    d_ktLimiter{centralUpwind::Limiter::vanAlbada};
    centralUpwind::FluxScheme d_ktScheme{centralUpwind::FluxScheme::Kurganov};
    centralUpwind::KappaMode  d_ktKappaMode{centralUpwind::KappaMode::MachToAcCourant};

    // Just the single ICE material's DW index.  computePressFC requires the
    // ICE-material ktAlphaOwn_FC* while press_CC/press_FC are pressure-material
    // variables, so it needs its own out-of-domain subset.
    MaterialSubset * d_ktICE_matl{nullptr};

    //__________________________________
    //  PITCH outer-loop options, <CFD><SPICE><PITCH>
    int    d_pitchOuter{1};       // nOuterCorrectors; 1 = PITCH off
    int    d_pitchNCorr{1};       // nCorrectors (impICE only, future)
    double d_pitchTheta{0.5};     // Crank-Nicolson weight
    double d_pitchTol{0.0};       // 0 = fixed iteration count

    // Re-runs the fluid chain once per outer iteration.  Created in
    // problemSetup only when d_pitchOuter > 1.
    SchedulerP d_pitchSched;
    bool       d_recompilePitchSched{true};

    // One "<tvar>_adv0" label per transported variable (see
    // pitchTvarAdv0Label()).
    std::vector<const VarLabel*> d_pitchTvarAdv0Labels;

    //__________________________________
    //  FSPICE: MPM materials allowed, and the PITCH hooks (setPitchHooks)
    bool d_hostedByFSPICE{false};
    std::function<void(SchedulerP&, const LevelP&)> d_pitchChainHook;
    std::function<void(Task*)>                       d_pitchRequiresHook;

    friend class FSPICE;
  };

} // End namespace Uintah

#endif
