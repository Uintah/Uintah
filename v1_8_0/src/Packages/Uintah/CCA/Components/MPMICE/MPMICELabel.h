#ifndef UINTAH_HOMEBREW_MPMICELABEL_H
#define UINTAH_HOMEBREW_MPMICELABEL_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <vector>

namespace Uintah {
using std::vector;

    class MPMICELabel {
    public:

      MPMICELabel();
      ~MPMICELabel();

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* vel_CCLabel;
      const VarLabel* vel_CC_scratchLabel;
      const VarLabel* temp_CCLabel;
      const VarLabel* temp_CC_scratchLabel;  // needed in doCCMomExchange()
      const VarLabel* press_NCLabel;
      const VarLabel* velInc_CCLabel;
      const VarLabel* velInc_NCLabel;
      const VarLabel* burnedMassCCLabel;
      const VarLabel* scratchLabel;         // to vis intermediate quantities
      const VarLabel* scratch1Label;
      const VarLabel* scratch2Label;
      const VarLabel* scratch3Label; 
      const VarLabel* scratchVecLabel;
      const VarLabel* NC_CCweightLabel;
      const VarLabel* rho_CCScratchLabel;
    };

} // end namespace Uintah

#endif
