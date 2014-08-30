/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef EXTERNALSTENCILTEST_H_
#define EXTERNALSTENCILTEST_H_
#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/SpatialField.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/structured/IndexTriplet.h>

namespace SpatialOps {
  namespace Point {

    struct PointFieldTraits {
      typedef NODIR FaceDir;
      typedef IndexTriplet< 0, 0, 0> Offset;
      typedef IndexTriplet<0,0,0>  BCExtra;
    };

    /**
     *  \brief The PointField type is intended for use in extracting and
     *         working with individual points from within another field type.
     *
     *  This field type is not compatible with operations such as
     *  interpolants, gradients, etc.  Operators are provided to extract
     *  points from a parent field and return them back to a parent
     *  field.
     */
    typedef SpatialField<Point::PointFieldTraits, float> PointFloatField;

  } // namespace Point
} // namespace SpatialOps



#endif /* EXTERNALSTENCILTEST_H_ */
