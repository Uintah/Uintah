
#ifndef UINTAH_HOMEBREW_CellInformationP_H
#define UINTAH_HOMEBREW_CellInformationP_H

namespace Uintah {
   template<class T> class Handle;
   namespace ArchesSpace {
      class CellInformation;
      typedef Handle<CellInformation> CellInformationP;
   }
}

#endif
