/**
 *  \file   IntVec.h
 *
 *  \date   Sep 28, 2011
 *  \author James C. Sutherland
 *
 *
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

#ifndef SpatialOps_IntVec_h
#define SpatialOps_IntVec_h

#include <spatialops/structured/Numeric3Vec.h>

namespace SpatialOps{

  typedef  Numeric3Vec<int> IntVec; ///< a lightweight class to deal with a 3D vector of integers.

  /**
   * @param dim the 3D layout of points
   * @param loc the 3D index
   * @return the 1D (flat) index
   * Assumes a fortran-style layout (e.g. first index varies fastest)
   */
  inline int ijk_to_flat( const IntVec& dim, const IntVec& loc ){
#   ifndef NDEBUG
    assert( loc[0] < dim[0] && loc[1] < dim[1] && loc[2] < dim[2] );
    assert( loc[0] >= 0 && loc[1] >= 0 && loc[2] >= 0 );
#   endif
    return loc[0] + loc[1]*dim[0] + loc[2]*dim[0]*dim[1];
  }

  /**
   * @param dim the 3D layout of points
   * @param pt the 1d (flat) index
   * @return the 3D index
   * Assumes a fortran-style layout (e.g. first index varies fastest)
   */
  inline IntVec flat_to_ijk( const IntVec& dim, const int pt ){
#   ifndef NDEBUG
    assert( pt >= 0 );
    assert( pt < dim[0]*dim[1]*dim[2] );
#   endif
    return IntVec( pt % dim[0], (pt / dim[0]) % dim[1], pt / (dim[0]*dim[1]) );
  }

  inline IntVec min(const IntVec& first, const IntVec& second) {
      return IntVec((first[0] < second[0] ? first[0] : second[0]),
                    (first[1] < second[1] ? first[1] : second[1]),
                    (first[2] < second[2] ? first[2] : second[2]));
  }

} // namespace SpatialOps

#endif /* SpatialOps_IntVec_h */
