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

      const VarLabel* cMassLabel;
      const VarLabel* cVolumeLabel;
      const VarLabel* vel_CCLabel;
      const VarLabel* mom_L_CCLabel;
      const VarLabel* dvdt_CCLabel;

      const VarLabel* fVelocityLabel;
      const VarLabel* fMassLabel;
    };
  } // end namepsace MPMICE
} // end namespace Uintah


// $Log$
// Revision 1.4  2001/01/15 23:21:54  guilkey
// Cleaned up CCMomentum exchange, so it now looks more like Todd's.
// Added effects back to solid material.  Need NodeIterator to be fixed,
// and need to figure out how to apply BCs from the ICE code.
//
// Revision 1.3  2001/01/14 02:30:01  guilkey
// CC momentum exchange now works from solid to fluid, still need to
// add fluid to solid effects.
//
// Revision 1.2  2001/01/11 20:11:16  guilkey
// Working on getting momentum exchange to work.  It doesnt' yet.
//
// Revision 1.1  2000/12/28 20:26:36  guilkey
// More work on coupling MPM and ICE
//

#endif
