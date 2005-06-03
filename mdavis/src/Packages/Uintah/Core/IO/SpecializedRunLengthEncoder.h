#ifndef UINTAH_SPECIALIZEDRUNLENGTHENCODER_H
#define UINTAH_SPECIALIZEDRUNLENGTHENCODER_H

// RunLengthEncoder with speciailizations for basic types, Vector's,
// and Point's. 
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Geometry/SpecializedRunLengthEncoder.h>

// Specializations of the DefaultRunLengthSequencer, to override the
// EqualElementSequencer version with the EqualIntervalSequencer

namespace SCIRun {
  using namespace Uintah;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1375
#endif
  
  // specialize for the Matrix3 class
  template<>
  class DefaultRunLengthSequencer<Matrix3>
    : public EqualIntervalSequencer<Matrix3>
  { };

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

}

#endif

