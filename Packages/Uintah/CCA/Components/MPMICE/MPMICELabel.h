#ifndef UINTAH_HOMEBREW_MPMICELABEL_H
#define UINTAH_HOMEBREW_MPMICELABEL_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {

    class MPMICELabel {
    public:

      MPMICELabel();
      ~MPMICELabel();

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* vel_CCLabel;
      const VarLabel* vel_CC_scratchLabel;
      const VarLabel* dvdt_CCLabel;
      const VarLabel* dTdt_CCLabel;
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
      const VarLabel* NC_CCweightLabel;
    };

} // end namespace Uintah

#endif
