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

#include <cstdint>
#include <type_traits>

namespace Uintah {

class BlockRange
{
public:

  enum { rank = 3 };

  template <typename ArrayType>
  BlockRange( ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  int begin( int r ) const { return m_offset[r]; }
  int   end( int r ) const { return m_offset[r] + m_dim[r]; }

  size_t size() const
  {
    size_t result = 1u;
    for (int i=0; i<rank; ++i) {
      result *= m_dim[i];
    }
    return result;
  }

  // populate the multi-index associated with the given linear index
  //   0 <= x < size()
  //   range(x, i, j, k);
  template <typename... Indices>
  inline __attribute__((always_inline))
  void operator()(int64_t x, Indices &... indices) const
  {
    static_assert( sizeof...(Indices) == rank
                  ,"Error: The number of indices does not equal the rank" );
    apply(0, x, indices...);
  }

private:

  template <typename... Indices>
  inline __attribute__((always_inline))
  void apply(int d, int64_t x, int & idx, Indices &... indices) const
  {
    idx = (x % m_dim[d]) + m_offset[d];
    apply( d+1, x/m_dim[d], indices...);
  }

  inline __attribute__((always_inline))
  void apply(int d, int64_t x, int & idx) const
  {
    idx = x + m_offset[d];
  }

private:
  int m_offset[rank];
  int m_dim[rank];
};


template <typename Functor>
void parallel_for( BlockRange const & r, const Functor & f )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

#if defined( UINTAH_ENABLE_KOKKOS )
#pragma omp parallel for collapse(3)
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
#else
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
#endif
};

template <typename Functor, typename ReductionType>
void parallel_reduce( BlockRange const & r, const Functor & f, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

ReductionType tmp = red;
#if defined( UINTAH_ENABLE_KOKKOS )
#pragma omp parallel for collapse(3) reduction(+:tmp)
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k,tmp);
  }}}
#else
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k,tmp);
  }}}
#endif
  red = tmp;
};

} // namespace Uintah

#endif // UINTAH_HOMEBREW_BLOCK_RANGE_H
