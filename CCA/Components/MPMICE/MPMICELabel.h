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
      const VarLabel* mom_L_CCLabel;
      const VarLabel* dvdt_CCLabel;

      const VarLabel* fVelocityLabel;
      const VarLabel* fMassLabel;
    };

} // end namespace Uintah

#endif
