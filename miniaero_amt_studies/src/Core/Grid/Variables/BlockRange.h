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

namespace Uintah {

/// RowMajor mapping of [0,n) -> [ (i0,j0,k0,..), (i1,j1,k1,...) )
template <  typename SizeType = int
          , int Rank = 3
         >
class RowMajorRange
{
  static_assert( std::is_integral<SizeType>::value || std::is_enum<SizeType>::value
                ,"Error: SizeType must be an integral type"
               );

  static_assert( Rank > 0
                ,"Error: Rank must be greater than 0"
               );
public:

  using size_type = SizeType;
  static constexpr int rank = Rank;

  template <typename ArrayType>
  RowMajorRange( ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  size_type size() const
  {
    size_type result{1};
    for (int i=0; i<rank; ++i) {
      result *= m_dim[i];
    }
    return result;
  }

  // populate the multi-index associated with the given linear index
  // e.g.
  //   x is the linear index
  //   range(x, i, j, k);
  template <typename IType, typename... Indices>
  inline __attribute__((always_inline))
  void operator()(size_type x, IType & idx, Indices &... indices) const
  {
    static_assert( (sizeof...(Indices) == (rank-1))
                  ,"Error: The number of indices does not equal the rank" );

    static_assert( std::is_integral<IType>::value
                  ,"Error: Index is not a intergral type" );
    static_assert( !std::is_signed<size_type>::value || std::is_signed<IType>::value
                  ,"Error: Signed size_type requires using signed indices" );

    x = apply(1, x, indices...);
    idx = x + m_offset[0];
  }

private:

  template <typename IType, typename... Indices>
  inline __attribute__((always_inline))
  int apply(int d, size_type x, IType & idx, Indices &... indices) const
  {
    static_assert( std::is_integral<IType>::value
                  ,"Error: Index is not a intergral type" );
    static_assert( !std::is_signed<size_type>::value || std::is_signed<IType>::value
                  ,"Error: Signed size_type requires using signed indices" );

    x = apply( d+1, x, indices...);
    idx = (x % m_dim[d]) + m_offset[d];
    return x / m_dim[d];
  }

  template <typename IType>
  inline __attribute__((always_inline))
  int apply(int d, size_type x, IType & idx) const
  {
    static_assert( std::is_integral<IType>::value
                  ,"Error: Index is not a intergral type" );
    static_assert( !std::is_signed<size_type>::value || std::is_signed<IType>::value
                  ,"Error: Signed size_type requires using signed indices" );

    idx = (x % m_dim[d]) + m_offset[d];
    return x / m_dim[d];
  }

public:
  size_type m_offset[rank];
  size_type m_dim[rank];
};


/// ColumnMajor mapping of [0,n) -> [ (i0,j0,k0,..), (i1,j1,k1,...) )
template <  typename SizeType = int
          , int Rank = 3
         >
class ColumnMajorRange
{
  static_assert( std::is_integral<SizeType>::value || std::is_enum<SizeType>::value
                ,"Error: SizeType must be an integral type"
               );
public:

  using size_type = SizeType;
  static constexpr int rank = Rank;

  template <typename ArrayType>
  ColumnMajorRange( ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  size_type size() const
  {
    size_type result{1};
    for (int i=0; i<rank; ++i) {
      result *= m_dim[i];
    }
    return result;
  }

  // populate the multi-index associated with the given linear index
  // e.g.
  //   x is the linear index
  //   range(x, i, j, k);
  template <typename... Indices>
  inline __attribute__((always_inline))
  void operator()(size_type x, Indices &... indices) const
  {
    static_assert( sizeof...(Indices) == rank
                  ,"Error: The number of indices does not equal the rank" );
    apply(0, x, indices...);
  }

private:

  template <typename IType, typename... Indices>
  inline __attribute__((always_inline))
  void apply(int d, size_type x, IType & idx, Indices &... indices) const
  {
    static_assert( std::is_integral<IType>::value
                  ,"Error: Index is not a intergral type" );
    static_assert( !std::is_signed<size_type>::value || std::is_signed<IType>::value
                  ,"Error: Signed size_type requires using signed indices" );

    idx = (x % m_dim[d]) + m_offset[d];
    apply( d+1, x/m_dim[d], indices...);
  }

  template <typename IType>
  inline __attribute__((always_inline))
  void apply(int d, size_type x, IType & idx) const
  {
    static_assert( std::is_integral<IType>::value
                  ,"Error: Index is not a intergral type" );
    static_assert( !std::is_signed<size_type>::value || std::is_signed<IType>::value
                  ,"Error: Signed size_type requires using signed indices" );

    idx = x + m_offset[d];
  }

public:
  size_type m_offset[rank];
  size_type m_dim[rank];
};


template <typename SizeType, typename Functor>
void parallel_for( const RowMajorRange<SizeType,3> & r, const Functor & f )
{
  const SizeType ib = r.m_offset[0];
  const SizeType jb = r.m_offset[1];
  const SizeType kb = r.m_offset[2];

  const SizeType ie = ib + r.m_dim[0];
  const SizeType je = jb + r.m_dim[1];
  const SizeType ke = kb + r.m_dim[2];

#pragma omp parallel for collapse(3)
  for (SizeType i=ib; i<ie; ++i) {
  for (SizeType j=jb; j<je; ++j) {
  for (SizeType k=kb; k<ke; ++k) {
    f(i,j,k);
  }}}
};

template <typename SizeType, typename Functor>
void parallel_for( const ColumnMajorRange<SizeType,3> & r, const Functor & f )
{
  const SizeType ib = r.m_offset[0];
  const SizeType jb = r.m_offset[1];
  const SizeType kb = r.m_offset[2];

  const SizeType ie = ib + r.m_dim[0];
  const SizeType je = jb + r.m_dim[1];
  const SizeType ke = kb + r.m_dim[2];

#pragma omp parallel for collapse(3)
  for (SizeType k=kb; k<ke; ++k) {
  for (SizeType j=jb; j<je; ++j) {
  for (SizeType i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
};


template <typename SizeType, typename Functor, typename ReductionType>
void parallel_reduce( const RowMajorRange<SizeType,3> & r, const Functor & f, ReductionType & red  )
{
  const SizeType ib = r.m_offset[0];
  const SizeType jb = r.m_offset[1];
  const SizeType kb = r.m_offset[2];

  const SizeType ie = ib + r.m_dim[0];
  const SizeType je = jb + r.m_dim[1];
  const SizeType ke = kb + r.m_dim[2];

#pragma omp parallel for collapse(3) reduction(+:red)
  for (SizeType i=ib; i<ie; ++i) {
  for (SizeType j=jb; j<je; ++j) {
  for (SizeType k=kb; k<ke; ++k) {
    f(i,j,k,red);
  }}}
};

template <typename SizeType, typename Functor, typename ReductionType>
void parallel_reduce( const ColumnMajorRange<SizeType,3> & r, const Functor & f, ReductionType & red  )
{
  const SizeType ib = r.m_offset[0];
  const SizeType jb = r.m_offset[1];
  const SizeType kb = r.m_offset[2];

  const SizeType ie = ib + r.m_dim[0];
  const SizeType je = jb + r.m_dim[1];
  const SizeType ke = kb + r.m_dim[2];

#pragma omp parallel for collapse(3) reduction(+:red)
  for (SizeType k=kb; k<ke; ++k) {
  for (SizeType j=jb; j<je; ++j) {
  for (SizeType i=ib; i<ie; ++i) {
    f(i,j,k,red);
  }}}
};

} // namespace Uintah

#endif // UINTAH_HOMEBREW_BLOCK_RANGE_H
