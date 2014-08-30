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

/**
 *  \file   DoubleVec.h
 */
#ifndef SpatialOps_DoubleVec_h
#define SpatialOps_DoubleVec_h

#include <spatialops/structured/Numeric3Vec.h>

#include <cmath>
#include <limits>


namespace SpatialOps{
  /**
   * \typedef DoubleVec
   * \brief Defines a three-component vector of doubles useful for expressing things like coordinates.
   */
  typedef SpatialOps::Numeric3Vec<double> DoubleVec;

  // specialize the comparison operators

  template<>
  inline bool DoubleVec::operator==( const DoubleVec& v ) const
  {
    const static double eps = 2.0*std::numeric_limits<double>::epsilon();
    return std::abs( v.ijk[0]-ijk[0] ) < eps
        && std::abs( v.ijk[1]-ijk[1] ) < eps
        && std::abs( v.ijk[2]-ijk[2] ) < eps;
  }

  template<>
  inline bool DoubleVec::operator!=(const DoubleVec& v) const{
    return !( *this == v );
  }

} // namespace SpatialOps


#endif /* SpatialOps_DoubleVec_h */
