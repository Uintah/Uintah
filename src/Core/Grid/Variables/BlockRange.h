/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_BLOCK_RANGE_H
#define UINTAH_HOMEBREW_BLOCK_RANGE_H

#include <type_traits>
#include <array>
#include <algorithm>

namespace Uintah {

/// RowMajor mapping of [0,n) -> [ (i0,j0,k0,..), (i1,j1,k1,...) )
template <  typename IntType = int
          , int Rank = 3
         >
class RowMajorRange
{
  static_assert( std::is_integral<IntType>::value || std::is_enum<IntType>::value
                ,"Error: IntType must be an integral type"
               );
public:

  using int_type = IntType;
  static constexpr int rank = Rank;
  using array_type = std::array<int_type, rank>;

  RowMajorRange( array_type const & c0, array_type const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = std::min( c0[i], c1[i] );
      m_dim[i] = std::max( c0[i], c1[i] ) - m_offset[i];
    }
  }

  int_type size() const
  {
    int_type result{1};
    for (auto d : m_dim) {
      result *= d;
    }
    return result;
  }

  array_type operator[]( int_type x ) const
  {
    array_type result;
    for (int i=rank-1; i > 0; --i) {
      result[i] = (x % m_dim[i]) + m_offset[i];
      x /= m_dim[i];
    }
    result[0] = x + m_offset[0];

    return result;
  }

private:

  array_type m_offset;
  array_type m_dim;
};


/// ColumnMajor mapping of [0,n) -> [ (i0,j0,k0,..), (i1,j1,k1,...) )
template <  typename IntType = int
          , int Rank = 3
         >
class ColumnMajorRange
{
  static_assert( std::is_integral<IntType>::value || std::is_enum<IntType>::value
                ,"Error: IntType must be an integral type"
               );
public:

  using int_type = IntType;
  static constexpr int rank = Rank;
  using array_type = std::array<int_type, rank>;

  ColumnMajorRange( array_type const & c0, array_type const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = std::min( c0[i], c1[i] );
      m_dim[i] = std::max( c0[i], c1[i] ) - m_offset[i];
    }
  }

  int_type size() const
  {
    int_type result{1};
    for (auto d : m_dim) {
      result *= d;
    }
    return result;
  }

  array_type operator[]( int_type x ) const
  {
    array_type result;
    for (int i=0; i < rank-1; ++i) {
      result[i] = (x % m_dim[i]) + m_offset[i];
      x /= m_dim[i];
    }
    result[rank-1] = x + m_offset[rank-1];

    return result;
  }

private:

  array_type m_offset;
  array_type m_dim;
};


} // namespace Uintah

#endif // UINTAH_HOMEBREW_BLOCK_RANGE_H
