#ifndef GEOMETRY_SPECIALIZEDRUNLENGTHENCODER_H
#define GEOMETRY_SPECIALIZEDRUNLENGTHENCODER_H

#include <Core/Containers/RunLengthEncoder.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

// Specializations of the DefaultRunLengthSequencer, to override the
// EqualElementSequencer version with the EqualIntervalSequencer

namespace SCIRun {
#ifdef __sgi
#pragma set woff 1375
#endif

  template<>
  class DefaultRunLengthSequencer<Vector>
    : public EqualIntervalSequencer<Vector>
  { };
  
  template<>
  class DefaultRunLengthSequencer<Point> :
    public EqualIntervalSequencer<Point, Vector>
  { };

#ifdef __sgi
#pragma reset woff 1375
#endif  
}

#endif
