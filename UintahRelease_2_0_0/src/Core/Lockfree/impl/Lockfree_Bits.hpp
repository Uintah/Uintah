/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef LOCKFREE_BITS_HPP
#define LOCKFREE_BITS_HPP

#include "Lockfree_Macros.hpp"
#include <atomic>
#include <array>
#include <iostream>

namespace Lockfree { namespace Impl {

//-----------------------------------------------------------------------------
/// count_trailing_zeros( integer_type i )
///
/// Returns the number of leading 0-bits in i, starting at the most significant
/// bit position.
///
/// If i is 0 return the number of bits in the integer type.

LOCKFREE_FORCEINLINE
constexpr
int count_trailing_zeros( unsigned i )
{
  return i ? __builtin_ctz( i ) : sizeof(unsigned) * CHAR_BIT ;
}
LOCKFREE_FORCEINLINE
constexpr
int count_trailing_zeros( unsigned long i )
{
  return i ? __builtin_ctzl( i ) : sizeof(unsigned long) * CHAR_BIT ;
}
LOCKFREE_FORCEINLINE
constexpr
int count_trailing_zeros( unsigned long long i )
{
  return i ? __builtin_ctzll( i ) : sizeof(unsigned long long) * CHAR_BIT ;
}

LOCKFREE_FORCEINLINE
constexpr
int count_trailing_zeros( unsigned short i )
{
  return count_trailing_zeros( static_cast<unsigned>(i) );
}

LOCKFREE_FORCEINLINE
constexpr
int count_trailing_zeros( unsigned char i )
{
  return count_trailing_zeros( static_cast<unsigned>(i) );
}

template <typename T>
LOCKFREE_FORCEINLINE
constexpr
T complement( T const& t )
{
  static_assert( std::is_integral<T>::value, " Must be integral type ");
  return static_cast<T>(~t);
}


//-----------------------------------------------------------------------------

template <typename BlockType, unsigned NumBlocks>
class Bitset
{
  static_assert( std::is_unsigned<BlockType>::value, "BlockType must be an unsigned integer type");
  static_assert( NumBlocks > 0u, "NumBlocks must be greater than 0");

  static constexpr int log2( BlockType n )
  { return n>1u ? 1 + log2( n >> 1 ) : 0; }

public:
  static constexpr int capacity = NumBlocks * sizeof(BlockType) * CHAR_BIT;

  using block_type = BlockType;

  static constexpr block_type block_size  = sizeof(block_type) * CHAR_BIT;
  static constexpr int        num_blocks  = capacity / block_size;
  static constexpr int        block_mask  = block_size - 1u;
  static constexpr int        block_shift = log2(block_size);
  static constexpr block_type one         = 1u;
  static constexpr block_type zero        = 0u;

  Bitset() = default;

  Bitset( const Bitset & ) = delete;
  Bitset & operator=( const Bitset & ) = delete;
  Bitset( Bitset && ) = delete;
  Bitset & operator=( Bitset && ) = delete;


  LOCKFREE_FORCEINLINE
  // return true if this call set the bit to 1
  bool set( int i, std::memory_order order = std::memory_order_relaxed )
  {
    const block_type bit = one << (i & block_mask);
    return !(m_blocks[ i >> block_shift ].fetch_or( bit, order) & bit);
  }

  LOCKFREE_FORCEINLINE
  // return true if this call set the bit to 0
  bool clear( int i, std::memory_order order = std::memory_order_relaxed )
  {
    const block_type bit = one << (i & block_mask);
    return m_blocks[ i >> block_shift ].fetch_and( complement(bit), order) & bit;
  }

  LOCKFREE_FORCEINLINE
  // return true if the ith bit is set
  bool test( int i, std::memory_order order = std::memory_order_relaxed ) const
  {
    const block_type bit = one << (i & block_mask);
    return m_blocks[ i >> block_shift ].load( order ) & bit;
  }

  LOCKFREE_FORCEINLINE
  block_type and_block( int i, block_type b, std::memory_order order = std::memory_order_relaxed )
  {
    return m_blocks[i].fetch_and( b, order );
  }

  LOCKFREE_FORCEINLINE
  block_type or_block( int i, block_type b, std::memory_order order = std::memory_order_relaxed )
  {
    return m_blocks[i].fetch_or( b, order );
  }

  LOCKFREE_FORCEINLINE
  block_type load_block( int i, std::memory_order order = std::memory_order_relaxed ) const
  {
    return m_blocks[i].load( order );
  }

  friend std::ostream & operator<<(std::ostream & out, Bitset const& b) {
    for (int i=0; i<b.num_blocks; ++i) {
      out << std::hex << b.load_block(i) << " ";
    }
    out << std::dec << "\b";
    return out;
  }

private:

  std::array< std::atomic<block_type>, num_blocks> m_blocks{};
};

}} // namespace Lockfree::Impl

#endif // LOCKFREE_BITS_HPP
