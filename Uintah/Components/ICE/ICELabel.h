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
    const VarLabel* press_CCLabel_0;
    const VarLabel* press_CCLabel_1;
    const VarLabel* press_CCLabel_2;
    const VarLabel* press_CCLabel_3;
    const VarLabel* press_CCLabel_4;
    const VarLabel* press_CCLabel_5;
    const VarLabel* press_CCLabel_6_7;

    const VarLabel* rho_CCLabel;
    const VarLabel* rho_CCLabel_0;
    const VarLabel* rho_CCLabel_1;
    const VarLabel* rho_CCLabel_2;
    const VarLabel* rho_CCLabel_3;
    const VarLabel* rho_CCLabel_4;
    const VarLabel* rho_CCLabel_5;
    const VarLabel* rho_CCLabel_6_7;

    const VarLabel* temp_CCLabel;
    const VarLabel* temp_CCLabel_0;
    const VarLabel* temp_CCLabel_1;
    const VarLabel* temp_CCLabel_2;
    const VarLabel* temp_CCLabel_3;
    const VarLabel* temp_CCLabel_4;
    const VarLabel* temp_CCLabel_5;
    const VarLabel* temp_CCLabel_6_7;

    const VarLabel* vel_CCLabel;
    const VarLabel* vel_CCLabel_0;
    const VarLabel* vel_CCLabel_1;
    const VarLabel* vel_CCLabel_2;
    const VarLabel* vel_CCLabel_3;
    const VarLabel* vel_CCLabel_4;
    const VarLabel* vel_CCLabel_5;
    const VarLabel* vel_CCLabel_6_7;

    const VarLabel* cv_CCLabel;
    const VarLabel* div_velfc_CCLabel;

    // Face centered variables
    const VarLabel* vel_FCLabel;
    const VarLabel* press_FCLabel;
    const VarLabel* tau_FCLabel;
      
    };
  } // end namepsace ICE
} // end namespace Uintah

#endif
// $Log$
// Revision 1.2  2000/10/04 20:17:52  jas
// Change namespace ICE to ICESpace.
//
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
