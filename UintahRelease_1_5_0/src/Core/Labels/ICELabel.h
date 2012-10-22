/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_ICELABEL_H
#define UINTAH_HOMEBREW_ICELABEL_H

#include <Core/Grid/Variables/VarLabel.h>


namespace Uintah {

  class ICELabel {
    public:

      ICELabel();
      ~ICELabel();

    const VarLabel* delTLabel;
    const VarLabel* NeedAddIceMaterialLabel;

    // Cell centered variables
    const VarLabel* TMV_CCLabel;
    const VarLabel* press_CCLabel;
    const VarLabel* press_equil_CCLabel;
    const VarLabel* delP_DilatateLabel;
    const VarLabel* delP_MassXLabel;
    const VarLabel* rho_micro_CCLabel;
    const VarLabel* sp_vol_CCLabel;
    const VarLabel* DLabel;
    const VarLabel* sum_rho_CCLabel;
    const VarLabel* compressibilityLabel;
    const VarLabel* sumKappaLabel;
    const VarLabel* rho_CCLabel;
    const VarLabel* temp_CCLabel;
    const VarLabel* temp_CC_XchangeLabel;
    const VarLabel* vel_CCLabel;
    const VarLabel* vel_CC_XchangeLabel;
    const VarLabel* speedSound_CCLabel;
    const VarLabel* vol_frac_CCLabel;
    const VarLabel* mom_source_CCLabel;
    const VarLabel* int_eng_source_CCLabel;
    const VarLabel* heatCond_src_CCLabel;
    const VarLabel* sp_vol_src_CCLabel;
    const VarLabel* mom_L_CCLabel;
    const VarLabel* int_eng_L_CCLabel;
    const VarLabel* sp_vol_L_CCLabel;
    const VarLabel* mass_L_CCLabel;
    const VarLabel* mom_L_ME_CCLabel;
    const VarLabel* eng_L_ME_CCLabel;
    const VarLabel* mass_advLabel;
    const VarLabel* mom_advLabel;
    const VarLabel* eng_advLabel;
    const VarLabel* sp_vol_advLabel;
    const VarLabel* created_vol_CCLabel;
    const VarLabel* term2Label;
    const VarLabel* term3Label;
    const VarLabel* f_theta_CCLabel;
    const VarLabel* Tdot_CCLabel;
    const VarLabel* turb_viscosity_CCLabel;
    const VarLabel* viscosityLabel;
    const VarLabel* thermalCondLabel;
    const VarLabel* gammaLabel;
    const VarLabel* specific_heatLabel;
    const VarLabel* dTdt_CCLabel;
    const VarLabel* dVdt_CCLabel;
//__________________________________
//    IMPLICIT      
    const VarLabel* matrixLabel;
    const VarLabel* rhsLabel;
    const VarLabel* initialGuessLabel;
    const VarLabel* betaLabel;
    const VarLabel* imp_delPLabel;
    const VarLabel* sum_imp_delPLabel;
    const VarLabel* sp_volX_FCLabel;
    const VarLabel* sp_volY_FCLabel;    
    const VarLabel* sp_volZ_FCLabel;
    const VarLabel* vol_fracX_FCLabel;
    const VarLabel* vol_fracY_FCLabel;    
    const VarLabel* vol_fracZ_FCLabel;
    
    // Face centered variables
    const VarLabel* uvel_FCLabel;
    const VarLabel* vvel_FCLabel;
    const VarLabel* wvel_FCLabel;
    const VarLabel* uvel_FCMELabel;
    const VarLabel* vvel_FCMELabel;
    const VarLabel* wvel_FCMELabel;
    const VarLabel* pressX_FCLabel;
    const VarLabel* pressY_FCLabel;
    const VarLabel* pressZ_FCLabel;
    const VarLabel* TempX_FCLabel;
    const VarLabel* TempY_FCLabel;
    const VarLabel* TempZ_FCLabel;
    const VarLabel* grad_P_XFCLabel;
    const VarLabel* grad_P_YFCLabel;
    const VarLabel* grad_P_ZFCLabel;
    const VarLabel* grad_dp_XFCLabel;
    const VarLabel* grad_dp_YFCLabel;
    const VarLabel* grad_dp_ZFCLabel;

    // Model support
    const VarLabel* modelMass_srcLabel;
    const VarLabel* modelMom_srcLabel;
    const VarLabel* modelEng_srcLabel;
    const VarLabel* modelVol_srcLabel;
    
    //Misc Labels
    const VarLabel* IveBeenHereLabel;
    const VarLabel* machLabel;
    const VarLabel* scratchLabel;
    const VarLabel* scratchVecLabel;
    const VarLabel* scratch_FCXLabel;
    const VarLabel* scratch_FCYLabel;
    const VarLabel* scratch_FCZLabel;
    
    // LODI boundary condition labels
    const VarLabel* LODI_BC_Li1Label;
    const VarLabel* LODI_BC_Li2Label;
    const VarLabel* LODI_BC_Li3Label;
    const VarLabel* LODI_BC_Li4Label;
    const VarLabel* LODI_BC_Li5Label;
    
    // Reduction Variables
    const VarLabel*  TotalMassLabel;
    const VarLabel*  TotalMomentumLabel;
    const VarLabel*  KineticEnergyLabel;
    const VarLabel*  TotalIntEngLabel;
    const VarLabel*  eng_exch_errorLabel;
    const VarLabel*  mom_exch_errorLabel;
    const VarLabel*  max_RHSLabel;
    
    const VarLabel*  maxMach_xminusLabel;
    const VarLabel*  maxMach_xplusLabel;
    const VarLabel*  maxMach_yminusLabel;
    const VarLabel*  maxMach_yplusLabel; 
    const VarLabel*  maxMach_zminusLabel;
    const VarLabel*  maxMach_zplusLabel;
    
    // AMR variables
    const VarLabel* AMR_SyncTaskgraphLabel;
    const VarLabel* mag_grad_rho_CCLabel;
    const VarLabel* mag_grad_temp_CCLabel;
    const VarLabel* mag_div_vel_CCLabel;
    const VarLabel* mag_grad_vol_frac_CCLabel;
    const VarLabel* mag_grad_press_CCLabel;
    
    // refluxing variables
    const VarLabel* mass_X_FC_fluxLabel;
    const VarLabel* mass_Y_FC_fluxLabel;
    const VarLabel* mass_Z_FC_fluxLabel;
    
    const VarLabel* mom_X_FC_fluxLabel;
    const VarLabel* mom_Y_FC_fluxLabel;
    const VarLabel* mom_Z_FC_fluxLabel;
    
    const VarLabel* sp_vol_X_FC_fluxLabel;
    const VarLabel* sp_vol_Y_FC_fluxLabel;
    const VarLabel* sp_vol_Z_FC_fluxLabel;

    const VarLabel* int_eng_X_FC_fluxLabel;
    const VarLabel* int_eng_Y_FC_fluxLabel;
    const VarLabel* int_eng_Z_FC_fluxLabel;
    
    const VarLabel* vol_frac_X_FC_fluxLabel;
    const VarLabel* vol_frac_Y_FC_fluxLabel;
    const VarLabel* vol_frac_Z_FC_fluxLabel;
    
    // face centered corrections computed in refluxing
    const VarLabel* mass_X_FC_corrLabel;
    const VarLabel* mass_Y_FC_corrLabel;
    const VarLabel* mass_Z_FC_corrLabel;
    
    const VarLabel* mom_X_FC_corrLabel;
    const VarLabel* mom_Y_FC_corrLabel;
    const VarLabel* mom_Z_FC_corrLabel;
    
    const VarLabel* sp_vol_X_FC_corrLabel;
    const VarLabel* sp_vol_Y_FC_corrLabel;
    const VarLabel* sp_vol_Z_FC_corrLabel;
    
    const VarLabel* int_eng_X_FC_corrLabel;
    const VarLabel* int_eng_Y_FC_corrLabel;
    const VarLabel* int_eng_Z_FC_corrLabel;
    
    const VarLabel* vol_frac_X_FC_corrLabel;
    const VarLabel* vol_frac_Y_FC_corrLabel;
    const VarLabel* vol_frac_Z_FC_corrLabel;

    // Implicit AMR variables
    const VarLabel* matrix_CFI_weightsLabel;

    };
} // end namespace Uintah

#endif
