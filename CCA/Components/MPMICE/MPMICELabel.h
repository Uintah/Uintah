#ifndef UINTAH_HOMEBREW_MPMICELABEL_H
#define UINTAH_HOMEBREW_MPMICELABEL_H

#include <Packages/Uintah/Core/Grid/VarLabel.h>

namespace Uintah {

    class MPMICELabel {
    public:

      MPMICELabel();
      ~MPMICELabel();

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* vel_CCLabel;
      const VarLabel* temp_CCLabel;
      const VarLabel* press_NCLabel;
      const VarLabel* burnedMassCCLabel;
      const VarLabel* onSurfaceLabel;
      const VarLabel* surfaceTempLabel;
      const VarLabel* scratchLabel;         // to vis intermediate quantities
      const VarLabel* scratch1Label;
      const VarLabel* scratch2Label;
      const VarLabel* scratch3Label; 
      const VarLabel* scratchVecLabel;
      const VarLabel* NC_CCweightLabel;
 
      const VarLabel* TempGradLabel;      // Needed by burn model --- temporary 
      const VarLabel* aveSurfTempLabel;    
    };

} // end namespace Uintah

#endif
