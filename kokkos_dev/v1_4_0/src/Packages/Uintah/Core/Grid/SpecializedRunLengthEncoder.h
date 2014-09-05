#ifndef UINTAH_SPECIALIZEDRUNLENGTHENCODER_H
#define UINTAH_SPECIALIZEDRUNLENGTHENCODER_H

// RunLengthEncoder with speciailizations for basic types, Vector's,
// and Point's. 
#include <Core/Geometry/SpecializedRunLengthEncoder.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>

// Specializations of the DefaultRunLengthSequencer, to override the
// EqualElementSequencer version with the EqualIntervalSequencer

namespace SCIRun {
  using namespace Uintah;

#ifdef __sgi
#pragma set woff 1375
#endif
  
  // specialize for the Matrix3 class
  template<>
  class DefaultRunLengthSequencer<Matrix3>
    : public EqualIntervalSequencer<Matrix3>
  { };

#ifdef __sgi
#pragma reset woff 1375
#endif

}

#endif

