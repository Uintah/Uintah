
#ifndef Packages_Uintah_Core_Grid_Stencil7_h
#define Packages_Uintah_Core_Grid_Stencil7_h

namespace Uintah {
  class TypeDescription;
  struct Stencil7 {
    // diagonal term
    double p;
    //     +x -x +y -y +z -z
    double  e, w, n, s, t, b;
  };
}

namespace SCIRun {
  void swapbytes( Uintah::Stencil7& );
} // namespace SCIRun

#endif
