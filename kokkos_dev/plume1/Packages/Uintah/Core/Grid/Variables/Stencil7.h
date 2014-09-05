
#ifndef Packages_Uintah_Core_Grid_Stencil7_h
#define Packages_Uintah_Core_Grid_Stencil7_h

#include <Core/Util/FancyAssert.h>

namespace Uintah {
  class TypeDescription;
  struct Stencil7 {
    // The order of this is designed to match the order of faces in Patch
    // Do not change it!
    //     -x +x -y +y -z +z
    double  w, e, s, n, b, t;
    // diagonal term
    double p;
    double& operator[](int index) {
      ASSERTRANGE(index, 0, 7);
      return (&w)[index];
    }
    const double& operator[](int index) const {
      ASSERTRANGE(index, 0, 7);
      return (&w)[index];
    }
  };
}

namespace SCIRun {
  void swapbytes( Uintah::Stencil7& );
} // namespace SCIRun

#endif
