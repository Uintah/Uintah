#ifndef ExplicitTimeInt_h
#define ExplicitTimeInt_h

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

#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>

#ifdef DO_TIMINGS
#  include <spatialops/util/TimeLogger.h>
#endif

//===========================================================================

namespace Uintah {

class ArchesLabel;
class ExplicitTimeInt {

public:

    ExplicitTimeInt(const ArchesLabel* fieldLabels);

    ~ExplicitTimeInt();
    /** @brief Input file interface and constant intialization */
    void problemSetup(const ProblemSpecP& params);

   /** @brief A template forward Euler update for a single
               variable for a single patch */
    template <class phiT, class constphiT>
    void singlePatchFEUpdate( const Patch* patch,
                              phiT& phi,
                              constphiT& RHS,
                              double dt,
                              const std::string eqnName);
   /** @brief A template forward Euler update for a single
               variable for a single patch */
    template <class phiT, class constphiT>
    void singlePatchFEUpdate( const Patch* patch,
                              phiT& phi, constCCVariable<double>& old_den,
                              constCCVariable<double>& new_den,
                              constphiT& RHS,
                              double dt,
                              const std::string eqnName );

    /** @brief A template for time averaging using a Runge-kutta form without explicit density and no clipping */
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch,
                     phiT& phi,
                     constphiT& old_phi,
                     const int step );

    /** @brief A template for time averaging using a Runge-kutta form without explicit density*/
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch,
                     phiT& phi,
                     constphiT& old_phi,
                     const int step,
                     const double clip_tol,
                     const bool do_low_clip,  const double low_clip,
                     const bool do_high_clip, const double high_clip,
                     constCCVariable<double>& vol_fraction );


    /** @brief A template for time averaging using a Runge-kutta form for weighted abscissa*/
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch,
                     phiT& phi,
                     constphiT& old_phi,
                     const int step,
                     const double clip_tol,
                     const bool do_low_clip,  const double low_clip,
                     const bool do_high_clip, const double high_clip, constCCVariable<double>& weight,
                     constCCVariable<double>& vol_fraction );


    /** @brief A template for time averaging using a Runge-kutta form with density */
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch,
                     phiT& phi,
                     constphiT& old_phi,
                     constphiT& old_den,
                     constphiT& new_den,
                     const int step,
                     const double clip_tol,
                     const bool do_low_clip,  const double low_clip,
                     const bool do_high_clip, const double high_clip );

    /** @brief A task interface to the singlePatchFEUpdate */
    void sched_fe_update( SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         std::vector<std::string> phi,
                         std::vector<std::string> rhs,
                         int rkstep);

    void fe_update( const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    std::vector<std::string> phi_lab,
                    std::vector<std::string> rhs_lab,
                    int rkstep);

    /** @brief A task interface to the timeAvePhi */
    void sched_time_ave( SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         std::vector<std::string> phi,
                         int rkstep );

    void time_ave( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   std::vector<std::string> phi_lab,
                   int rkstep );

    Vector ssp_beta, ssp_alpha;
    Vector time_factor;

    double d_LinfError;
    double d_LinfSol;

    std::string d_time_order;

private:
    const ArchesLabel* d_fieldLabels;
    int d_step;

  }; //end Class ExplicitTimeInt

  // no density
  template <class phiT, class constphiT>
  void ExplicitTimeInt::singlePatchFEUpdate( const Patch* patch,
                                             phiT& phi,
                                             constphiT& RHS,
                                             double dt,
                                             const std::string eqnName)
  {

    Vector dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      phi[c] += dt/vol*(RHS[c]);

    }
  }

  // with density
  template <class phiT, class constphiT>
  void ExplicitTimeInt::singlePatchFEUpdate( const Patch* patch,
                                             phiT& phi, constCCVariable<double>& old_den,
                                             constCCVariable<double>& new_den,
                                             constphiT& RHS,
                                             double dt,
                                             const std::string eqnName )

  {

    Vector dx = patch->dCell();
    double dtvol = dt/ (dx.x()*dx.y()*dx.z());

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      // (rho*phi)^{t+\Delta t} = (rho*phi)^{t} + RHS
      phi[c] = old_den[c]*phi[c] + dtvol*(RHS[c]);

      // phi^{t+\Delta t} = ((rho*phi)^{t} + RHS) / rho^{t + \Delta t}
      //double rho_ox = .5;
      //double rho_f = 1.18;
      //double rho_guess = rho_ox + phi[c]*(1-rho_ox/rho_f);
      //phi[c] = phi[c] / rho_guess;
      phi[c] = phi[c] / new_den[c];

      double small = 1e-16;
      if (new_den[c] < small)
        phi[c] = 0.0;

    }
  }

//---------------------------------------------------------------------------
// Time averaging W/O explicit density and no clipping
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged phi^{time averaged}
//     See: Gottlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch,
                                    phiT& phi,
                                    constphiT& old_phi,
                                    const int step )
  {
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      phi[*iter] = ssp_alpha[step] * old_phi[c] + ssp_beta[step] * phi[c];

    }
  }
//---------------------------------------------------------------------------
// Time averaging W/O density
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged phi^{time averaged}
//     See: Gottlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch,
                                    phiT& phi,
                                    constphiT& old_phi,
                                    const int step,
                                    const double clip_tol,
                                    const bool do_low_clip,  const double low_clip,
                                    const bool do_high_clip, const double high_clip,
                                    constCCVariable<double>& vol_fraction )
  {

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      // Turning off clipping here. Clipping should be occuring upstream
      // based on other clipping mechanisms.
      // if ( do_low_clip && phi[c] < ( low_clip + clip_tol ) ){
      //
      //   phi[c] = low_clip * vol_fraction[c];
      //
      // } else if ( do_high_clip && phi[c] > ( high_clip + clip_tol ) ){
      //
      //   phi[c] = high_clip * vol_fraction[c];
      //
      // } else {
      //
      //   phi[c] = ssp_alpha[step] * old_phi[c] + ssp_beta[step] * phi[c];
      //
      // }

      phi[c] = ssp_alpha[step] * old_phi[c] + ssp_beta[step] * phi[c];

    }
  }


//---------------------------------------------------------------------------
// Time averaging W/O density
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged weighted phi^{time averaged}
//     See: Gottlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch,
                                    phiT& phi,
                                    constphiT& old_phi,
                                    const int step,
                                    const double clip_tol,
                                    const bool do_low_clip,  const double low_clip,
                                    const bool do_high_clip, const double high_clip,
                                    constCCVariable<double>& weight, constCCVariable<double>& vol_fraction )
  {

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      // Turning off clipping here. Clipping should be occuring upstream
      // based on other clipping mechanisms.
      // if (weight[c] == 0) {
      //
      //   phi[c] = 0.0;
      //
      // } else if ( do_low_clip && phi[c]/weight[c] < ( low_clip + clip_tol ) ){
      //
      //   phi[c] = weight[c] * low_clip * vol_fraction[c];
      //
      // } else if ( do_high_clip && phi[c]/weight[c] > ( high_clip - clip_tol ) ){
      //
      //   phi[c] = high_clip*weight[c] * vol_fraction[c];
      //
      // //} else {
      //
      //   phi[c] = ssp_alpha[step] * old_phi[c] + ssp_beta[step] * phi[c];
      //
      //}

      phi[c] = ssp_alpha[step] * old_phi[c] + ssp_beta[step] * phi[c];

    }
  }


//---------------------------------------------------------------------------
// Time averaging With Density
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged phi^{time averaged}
//     See: Gettlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch,
                                    phiT& phi,
                                    constphiT& old_phi,
                                    constphiT& new_den,
                                    constphiT& old_den,
                                    int step,
                                    const double clip_tol,
                                    const bool do_low_clip,  const double low_clip,
                                    const bool do_high_clip, const double high_clip )
  {


#ifdef DO_TIMINGS
    SpatialOps::TimeLogger timer("old_scalar_fe_update.out");
    timer.start("work");
#endif

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      double pred_density = ssp_alpha[step]*old_den[c] + ssp_beta[step]*new_den[c];

      if ( pred_density > 0 ) {

        if ( do_high_clip && phi[c] > ( high_clip - clip_tol) ){

          phi[c] = high_clip;

        } else if ( do_low_clip && phi[c] < ( low_clip + clip_tol) ){

          phi[c] = low_clip;

        } else {

          phi[c] = ( ssp_alpha[step] * (old_den[c] * old_phi[c])
                       + ssp_beta[step]  * (new_den[c] * phi[c]) ) / pred_density;

        }
      }
    }
#ifdef DO_TIMINGS
    timer.stop("work");
#endif
  }

} //end namespace Uintah

#endif
