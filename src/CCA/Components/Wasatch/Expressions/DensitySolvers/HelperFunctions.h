/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef Wasatch_DensitySolverHelperFunctions_h
#define Wasatch_DensitySolverHelperFunctions_h

#include <vector>

namespace WasatchCore{

  // \brief computes the flat index of a square array given 
  //        a row, index, and number of rows.
  inline unsigned square_ij_to_flat( const unsigned nRows, 
                                     const unsigned iRow,
                                     const unsigned jCol )
  {
      assert(nRows > iRow);
      assert(nRows > jCol);
      return iRow*nRows + jCol;
  }

  template<typename T>
  std::vector<T> sub_vector(std::vector<T> const& vec, int m, int n) 
  {
    assert(m < vec.size());
    assert(n < vec.size());
    assert(m >= 0);
    assert(n >= 0);

    const auto first = vec.cbegin() + m;
    const auto last  = vec.cbegin() + n;
    
    return std::vector<T>(first, last);
  }

  template<typename T>
  std::vector<T> concatenate_vectors(std::vector<T> const vec1, std::vector<T> const vec2) 
  {
    std::vector<T> result;
    result.reserve( vec1.size() + vec2.size() );
    result.insert( result.end(), vec1.begin(), vec1.end() );
    result.insert( result.end(), vec2.begin(), vec2.end() );
  
    return result;
  }

}
#endif