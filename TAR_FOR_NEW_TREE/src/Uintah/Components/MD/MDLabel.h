#ifndef UINTAH_HOMEBREW_MDLABEL_H
#define UINTAH_HOMEBREW_MDLABEL_H


#include <Uintah/Grid/VarLabel.h>

namespace Uintah {
  namespace MD {
    class MDLabel {
    public:

      MDLabel();
      ~MDLabel();

      static const MDLabel* getLabels();

      const VarLabel* delTLabel;
      
      const VarLabel* atomMassLabel;
      const VarLabel* atomXLabel;

    };
  } // end namepsace MD
} // end namespace Uintah

#endif

// $Log$
// Revision 1.1  2000/06/10 04:10:35  tan
// Added MDLabel class.
//
