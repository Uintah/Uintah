/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_SPECIALIZEDRUNLENGTHENCODER_H
#define UINTAH_SPECIALIZEDRUNLENGTHENCODER_H

// RunLengthEncoder with speciailizations for basic types, Vector's,
// and Point's. 
#include <Core/Math/Matrix3.h>
#include <Core/Geometry/SpecializedRunLengthEncoder.h>

// Specializations of the DefaultRunLengthSequencer, to override the
// EqualElementSequencer version with the EqualIntervalSequencer

namespace SCIRun {

  // specialize for the Matrix3 class
  template<>
  class DefaultRunLengthSequencer<Uintah::Matrix3> : public EqualIntervalSequencer<Uintah::Matrix3> { };

}

#endif

