/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef GEOMETRY_SPECIALIZEDRUNLENGTHENCODER_H
#define GEOMETRY_SPECIALIZEDRUNLENGTHENCODER_H

#include <Core/Containers/RunLengthEncoder.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

// Specializations of the DefaultRunLengthSequencer, to override the
// EqualElementSequencer version with the EqualIntervalSequencer

namespace SCIRun {
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif  
}

#endif
