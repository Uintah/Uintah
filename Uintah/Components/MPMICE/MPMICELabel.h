#ifndef UINTAH_HOMEBREW_MPMICELABEL_H
#define UINTAH_HOMEBREW_MPMICELABEL_H


#include <Uintah/Grid/VarLabel.h>
#include <vector>

using std::vector;

namespace Uintah {
  namespace MPMICESpace {
    class MPMICELabel {
    public:

      MPMICELabel();
      ~MPMICELabel();

      const VarLabel* cMomentumLabel;
      const VarLabel* cVelocityMELabel;
      const VarLabel* cMassLabel;

      const VarLabel* fVelocityLabel;
      const VarLabel* fMassLabel;
    };
  } // end namepsace MPMICE
} // end namespace Uintah


// $Log$
// Revision 1.2  2001/01/11 20:11:16  guilkey
// Working on getting momentum exchange to work.  It doesnt' yet.
//
// Revision 1.1  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//

#endif
