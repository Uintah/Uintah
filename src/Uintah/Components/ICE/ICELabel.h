#ifndef UINTAH_HOMEBREW_ICELABEL_H
#define UINTAH_HOMEBREW_ICELABEL_H


#include <Uintah/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {
  namespace ICESpace {
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
  } // end namepsace ICE
} // end namespace Uintah

#endif
// $Log$
// Revision 1.15  2001/01/03 00:51:53  harman
// - added cflux, OFC, IFC, q_in_CF, q_out_CF
// - Advection operator now in 3D, not fully tested
// - A little house cleaning on #includes
// - Removed *_old from step6&7 except cv_old
//
// Revision 1.14  2000/11/28 03:50:28  jas
// Added {X,Y,Z}FCVariables.  Things still don't work yet!
//
// Revision 1.13  2000/11/14 04:02:11  jas
// Added getExtraCellIterator and things now appear to be working up to
// face centered velocity calculations.
//
// Revision 1.12  2000/10/25 23:12:17  guilkey
// Fixed step2, reorganized 6and7 just a little bit.
//
// Revision 1.11  2000/10/24 23:07:21  guilkey
// Added code for steps6and7.
//
// Revision 1.10  2000/10/20 23:58:55  guilkey
// Added part of advection code.
//
// Revision 1.9  2000/10/19 02:44:52  guilkey
// Added code for step5b.
//
// Revision 1.8  2000/10/18 21:02:17  guilkey
// Added code for steps 4 and 5.
//
// Revision 1.7  2000/10/17 04:13:25  jas
// Implement hydrostatic pressure adjustment as part of step 1b.  Still need
// to implement update bcs.
//
// Revision 1.6  2000/10/16 19:10:35  guilkey
// Combined step1e with step2 and eliminated step1e.
//
// Revision 1.5  2000/10/13 00:01:11  guilkey
// More work on ICE
//
// Revision 1.4  2000/10/09 22:37:01  jas
// Cleaned up labels and added more computes and requires for EOS.
//
// Revision 1.3  2000/10/05 04:26:48  guilkey
// Added code for part of the EOS evaluation.
//
// Revision 1.2  2000/10/04 20:17:52  jas
// Change namespace ICE to ICESpace.
//
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
