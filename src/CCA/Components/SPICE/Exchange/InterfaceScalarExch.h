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

#ifndef UINTAH_SPICE_EXCHANGE_INTERFACESCALAREXCH_H
#define UINTAH_SPICE_EXCHANGE_INTERFACESCALAREXCH_H

#include <CCA/Components/Models/MultiMatlExchange/Scalar.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <vector>

namespace Uintah {

  class ExchangeCoefficients;
  class VarLabel;

namespace ExchangeModels{

/*______________________________________________________________________
 InterfaceScalarExch: the scalar (Kashiwa) exchange model plus the gas-solid
 interface options of FSPICE (SPICE + MPM), from <CFD><SPICE><CentralUpwind>.
 SPICE swaps it in for ICE's ScalarExch when MPM materials and one of the
 options below are on; with all of them off it is ScalarExch.

   setParticleAware (<interfaceExchange> particles/conduction)
       no gas-MPM HEAT exchange in a cell without particles (sharp particle
       solid fraction solidVolFracP_CC below particleExchMin and not inside
       the solid by the node test), and no gas-MPM FACE momentum exchange
       between two particle-free cells.  Heat moves at the material
       interface; a cell without particles holds only the node-averaged
       smear of the solid's mass, and the stock 1e15 lock there cooled the
       gas next to a wall into a dense layer that KT spread over ~4 cells,
       and held its face velocity at the smear's, so gas could not refill
       the cell in front of a wall that cooled it.  Cell-centred momentum
       exchange stays stock: without the drag tying the smear to the gas,
       the smear's velocity change, mapped onto the front nodes, crushed the
       wall's front particle layer (T2, to 3.5-5% of its volume).
       noGasSolidHeatExch (<interfaceExchange> conduction): no gas-MPM heat
       exchange in any cell; SPICE::interfaceConduction moves the heat.

   setTangentialSlip (<interfaceShear> slip/model)
       in a cell without particles the gas-MPM momentum exchange acts only
       along the surface normal (grad solidVolFracP_CC); its tangential part
       is scaled by 1 - the particle-free weight.  Cut cells (gas and
       particles) keep the full lock.  With the stock lock the gas in the
       cell in front of a wall moves with the wall in every direction: a
       no-slip wall half a cell off the surface at any viscosity.
______________________________________________________________________*/

  class InterfaceScalarExch : public ScalarExch {

  public:
    InterfaceScalarExch( const ProblemSpecP     & prob_spec,
                         const MaterialManagerP & materialManager,
                         const bool               with_mpm );

    virtual ~InterfaceScalarExch();

    virtual void problemSetup( const ProblemSpecP & prob_spec );

    //  solidVolFracP_CC / solidInteriorP_CC (SPICELabel), stored with the
    //  ICE material and computed by FSPICE::computeSolidVolFracP
    void setSolidLabels( const VarLabel * solidVolFracP,
                         const VarLabel * solidInteriorP )
    {
      d_solidVolFracPLabel  = solidVolFracP;
      d_solidInteriorPLabel = solidInteriorP;
    };

    void setParticleAware( const double particleExchMin,
                           const bool   noGasSolidHeatExch )
    {
      d_particleAwareExch  = true;
      d_particleExchMin    = particleExchMin;
      d_noGasSolidHeatExch = noGasSolidHeatExch;
    };

    void setTangentialSlip( const double particleExchMin )
    {
      d_tangentialSlip  = true;
      d_particleExchMin = particleExchMin;
    };

    virtual void sched_AddExch_VelFC( SchedulerP           & sched,
                                      const PatchSet       * patches,
                                      const MaterialSubset * ice_matls,
                                      const MaterialSubset * mpm_matls,
                                      const MaterialSet    * all_matls,
                                      customBC_globalVars  * BC_globalVars,
                                      const bool recursion );

    virtual void addExch_VelFC( const ProcessorGroup * pg,
                                const PatchSubset    * patch,
                                const MaterialSubset * matls,
                                DataWarehouse        * old_dw,
                                DataWarehouse        * new_dw,
                                customBC_globalVars  * BC_globalVars,
                                const bool recursion );

    virtual void sched_AddExch_Vel_Temp_CC( SchedulerP           & sched,
                                            const PatchSet       * patches,
                                            const MaterialSubset * ice_matls,
                                            const MaterialSubset * mpm_matls,
                                            const MaterialSet    * all_matls,
                                            customBC_globalVars  * BC_globalVars );

    virtual void addExch_Vel_Temp_CC( const ProcessorGroup * pg,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                      DataWarehouse        * old_dw,
                                      DataWarehouse        * new_dw,
                                      customBC_globalVars  * BC_globalVars );

  private:
    //  1 where a cell holds no particles, 0 where it does, a linear ramp
    //  in between (see the .cc)
    constCCVariable<double> particleFree( const constCCVariable<double> & boxFrac,
                                          const constCCVariable<double> & interior,
                                          const Patch                   * patch,
                                          DataWarehouse                 * new_dw );

    template<class constSFC, class SFC>
    void vel_FC_exchange( CellIterator  iter,
                          IntVector     adj_offset,
                          int           numMatls,
                          FastMatrix  & K,
                          double        delT,
                          std::vector<constCCVariable<double> >& vol_frac_CC,
                          std::vector<constCCVariable<double> >& sp_vol_CC,
                          std::vector< constSFC>               & vel_FC,
                          std::vector< SFC >                   & sp_vol_FC,
                          std::vector< SFC >                   & vel_FCME,
                          const CCVariable<double>             * noParticles = nullptr,
                          const std::vector<bool>              * isMPM       = nullptr );

    ExchangeCoefficients * d_ifExchCoeff{nullptr};

    const VarLabel * d_solidVolFracPLabel{nullptr};
    const VarLabel * d_solidInteriorPLabel{nullptr};

    bool   d_particleAwareExch{false};
    double d_particleExchMin{0.1};
    bool   d_noGasSolidHeatExch{false};
    bool   d_tangentialSlip{false};
  };
}
}

#endif
