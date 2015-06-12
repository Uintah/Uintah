#ifndef LOCKFREE_BITS_HPP
#define LOCKFREE_BITS_HPP

#include "Lockfree_Macros.hpp"

namespace Lockfree { namespace Impl {

//-----------------------------------------------------------------------------
/// count_trailing_zeros( integer_type i )
///
/// Returns the number of leading 0-bits in i, starting at the most significant
/// bit position.
///
/// If i is 0 return the number of bits in the integer type.

LOCKFREE_FORCEINLINE
int count_trailing_zeros( unsigned i )
{
  enum { bits = sizeof(unsigned) * CHAR_BIT };
  return i ? __builtin_ctz( i ) : bits ;
}
LOCKFREE_FORCEINLINE
int count_trailing_zeros( unsigned long i )
{
  enum { bits = sizeof(unsigned long) * CHAR_BIT };
  return i ? __builtin_ctzl( i ) : bits ;
}
LOCKFREE_FORCEINLINE
int count_trailing_zeros( unsigned long long i )
{
  enum { bits = sizeof(unsigned long long) * CHAR_BIT };
  return i ? __builtin_ctzll( i ) : bits ;
}

LOCKFREE_FORCEINLINE
int count_trailing_zeros( unsigned short i )
{
  return count_trailing_zeros( static_cast<unsigned>(i) );
}

LOCKFREE_FORCEINLINE
int count_trailing_zeros( unsigned char i )
{
  return count_trailing_zeros( static_cast<unsigned>(i) );
}

//-----------------------------------------------------------------------------
/// popcount( integer_type i )
///
/// Returns the number of 1-bits in i.

LOCKFREE_FORCEINLINE
int popcount( unsigned i )
{
  return __builtin_popcount( i );
}
LOCKFREE_FORCEINLINE
int popcount( unsigned long i )
{
  return __builtin_popcountl( i );
}
LOCKFREE_FORCEINLINE
int popcount( unsigned long long i )
{
  return __builtin_popcountll( i );
}

LOCKFREE_FORCEINLINE
int popcount( unsigned short i )
{
  return popcount( static_cast<unsigned>(i) );
}

LOCKFREE_FORCEINLINE
int popcount( unsigned char i )
{
  return popcount( static_cast<unsigned>(i) );
}

//-----------------------------------------------------------------------------
LOCKFREE_FORCEINLINE
unsigned char complement( unsigned char i )
{
  return static_cast<unsigned char>(~i);
}

LOCKFREE_FORCEINLINE
unsigned short complement( unsigned short i )
{
  return static_cast<unsigned short>(~i);
}

LOCKFREE_FORCEINLINE
unsigned complement( unsigned i )
{
  return ~i;
}

LOCKFREE_FORCEINLINE
unsigned long complement( unsigned long i )
{
  return ~i;
}

LOCKFREE_FORCEINLINE
unsigned long long complement( unsigned long long i )
{
  return ~i;
}
//-----------------------------------------------------------------------------

}} // namespace Lockfree::Impl

#endif // LOCKFREE_BITS_HPP
