#ifndef UINTAH_HOMEBREW_SimulationTime_H
#define UINTAH_HOMEBREW_SimulationTime_H

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Util/FancyAssert.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

class Matrix3;

namespace Uintah {

   inline bool isFlat(const double&) {
      return true;
   }
   inline bool isFlat(const SCICore::Geometry::Vector&) {
      ASSERTEQ(sizeof(SCICore::Geometry::Vector), 3*sizeof(double));
      return true;
   }
   inline bool isFlat(const SCICore::Geometry::Point&) {
      ASSERTEQ(sizeof(SCICore::Geometry::Point), 3*sizeof(double));
      return true;
   }
   inline bool isFlat(const Matrix3&) {
      return false;
   }
} // end namespace Uintah

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1209
#endif

//
// $Log$
// Revision 1.1  2000/05/15 19:39:47  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//

#endif

