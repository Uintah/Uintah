#ifndef UINTAH_HOMEBREW_ICELABEL_H
#define UINTAH_HOMEBREW_ICELABEL_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>

namespace Uintah {

  class ICELabel {
    public:

      ICELabel();
      ~ICELabel();

    const VarLabel* delTLabel;
    const VarLabel* doMechLabel;

    // Cell centered variables
    const VarLabel* press_CCLabel;
    const VarLabel* press_equil_CCLabel;
    const VarLabel* matl_press_CCLabel;
    const VarLabel* delP_DilatateLabel;
    const VarLabel* delP_MassXLabel;
    const VarLabel* rho_micro_CCLabel;
    const VarLabel* sp_vol_CCLabel;
    const VarLabel* DLabel;
    const VarLabel* sum_rho_CCLabel;
    const VarLabel* rho_CCLabel;
    const VarLabel* temp_CCLabel;
    const VarLabel* vel_CCLabel;
    const VarLabel* speedSound_CCLabel;
    const VarLabel* vol_frac_CCLabel;
    const VarLabel* press_force_CCLabel;
    const VarLabel* mom_source_CCLabel;
    const VarLabel* int_eng_source_CCLabel;
    const VarLabel* spec_vol_source_CCLabel;
    const VarLabel* mom_L_CCLabel;
    const VarLabel* int_eng_L_CCLabel;
    const VarLabel* spec_vol_L_CCLabel;
    const VarLabel* mass_L_CCLabel;
    const VarLabel* mom_L_ME_CCLabel;
    const VarLabel* eng_L_ME_CCLabel;
    const VarLabel* burnedMass_CCLabel;
    const VarLabel* int_eng_comb_CCLabel;
    const VarLabel* created_vol_CCLabel;
    const VarLabel* mom_comb_CCLabel;
    const VarLabel* term2Label;
    const VarLabel* term3Label;
    const VarLabel* f_theta_CCLabel;
    const VarLabel* Tdot_CCLabel;
//__________________________________
//    IMPLICIT      
    const VarLabel* matrixLabel;
    const VarLabel* rhsLabel;
    const VarLabel* initialGuessLabel;
    const VarLabel* betaLabel;
    const VarLabel* imp_delPLabel;
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
    const VarLabel* press_diffX_FCLabel;
    const VarLabel* press_diffY_FCLabel;
    const VarLabel* press_diffZ_FCLabel;

    // Model support
    const VarLabel* modelMass_srcLabel;
    const VarLabel* modelMom_srcLabel;
    const VarLabel* modelEng_srcLabel;
    const VarLabel* modelVol_srcLabel;
    
    //Misc Labels
    const VarLabel* IveBeenHereLabel;
    const VarLabel* scratchLabel;
    const VarLabel* scratch_FCXLabel;
    const VarLabel* scratch_FCYLabel;
    const VarLabel* scratch_FCZLabel;
    const VarLabel* scratch_FCVectorLabel;

    // Reduction Variables
    const VarLabel*  TotalMassLabel;
    const VarLabel*  CenterOfMassVelocityLabel;
    const VarLabel*  KineticEnergyLabel;
    const VarLabel*  TotalIntEngLabel;
    const VarLabel*  eng_exch_errorLabel;
    const VarLabel*  mom_exch_errorLabel;
    const VarLabel*  max_RHSLabel;
      
    };
} // end namespace Uintah

#endif
