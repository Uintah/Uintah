#ifndef UINTAH_HOMEBREW_ICELABEL_H
#define UINTAH_HOMEBREW_ICELABEL_H


#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {

  class ICELabel {
    public:

      ICELabel();
      ~ICELabel();

    const VarLabel* delTLabel;

    // Cell centered variables
    const VarLabel* press_CCLabel;
    const VarLabel* pressdP_CCLabel;
    const VarLabel* delPress_CCLabel;
    const VarLabel* rho_micro_CCLabel;
    const VarLabel* rho_micro_equil_CCLabel;
    const VarLabel* rho_CCLabel;
    const VarLabel* temp_CCLabel;
    const VarLabel* uvel_CCLabel;
    const VarLabel* vvel_CCLabel;
    const VarLabel* wvel_CCLabel;
    const VarLabel* speedSound_CCLabel;
    const VarLabel* speedSound_equiv_CCLabel;
    const VarLabel* cv_CCLabel;
    const VarLabel* div_velfc_CCLabel;
    const VarLabel* vol_frac_CCLabel;
    const VarLabel* viscosity_CCLabel;
    const VarLabel* xmom_source_CCLabel;
    const VarLabel* ymom_source_CCLabel;
    const VarLabel* zmom_source_CCLabel;
    const VarLabel* int_eng_source_CCLabel;
    const VarLabel* xmom_L_CCLabel;
    const VarLabel* ymom_L_CCLabel;
    const VarLabel* zmom_L_CCLabel;
    const VarLabel* int_eng_L_CCLabel;
    const VarLabel* mass_L_CCLabel;
    const VarLabel* rho_L_CCLabel;
    const VarLabel* xmom_L_ME_CCLabel;
    const VarLabel* ymom_L_ME_CCLabel;
    const VarLabel* zmom_L_ME_CCLabel;
    const VarLabel* int_eng_L_ME_CCLabel;
    const VarLabel* IFS_CCLabel;
    const VarLabel* OFS_CCLabel;
    const VarLabel* IFE_CCLabel;
    const VarLabel* OFE_CCLabel;
    const VarLabel* q_CCLabel;
    const VarLabel* q_advectedLabel;
    const VarLabel* term1Label;
    const VarLabel* term2Label;
    const VarLabel* term3Label;
   
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
    const VarLabel* tau_X_FCLabel;
    const VarLabel* tau_Y_FCLabel;
    const VarLabel* tau_Z_FCLabel;
      
  };
} // End namespace Uintah

#endif
