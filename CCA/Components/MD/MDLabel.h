#ifndef UINTAH_HOMEBREW_MDLABEL_H
#define UINTAH_HOMEBREW_MDLABEL_H


#include <Packages/Uintah/Core/Grid/VarLabel.h>

namespace Uintah {
    public:

      MDLabel();
      ~MDLabel();

      static const MDLabel* getLabels();

      const VarLabel* delTLabel;
      
      const VarLabel* atomMassLabel;
      const VarLabel* atomXLabel;

} // End namespace Uintah
    };

#endif

