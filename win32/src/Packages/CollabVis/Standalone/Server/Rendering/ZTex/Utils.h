#ifndef _util_h
#define _util_h

#include <string.h>
#include <stdlib.h>

namespace SemotusVisum {
namespace Rendering {

// Swap endian-ness of the array. length is the number of 4-byte words to swap.
static inline void ConvertLong(unsigned int *array, int length)
{
  unsigned int b1, b2, b3, b4;
  unsigned char *ptr = (unsigned char *)array;

  while (length--) {
    b1 = *ptr++;
    b2 = *ptr++;
    b3 = *ptr++;
    b4 = *ptr++;
#ifdef SCI_LITTLE_ENDIAN
    *array++ = (unsigned int) ((b1 << 24) | (b2 << 16) | (b3 << 8) | (b4));
#else
    *array++ = (unsigned int) ((b1) | (b2 << 8) | (b3 << 16) | (b4 << 24));
#endif
  }
}

// Swap endian-ness of the array. length is the number of 2-byte shorts to swap.
static inline void ConvertShort(unsigned short *array, int length)
{
  unsigned int b1, b2;
  unsigned char *ptr;
  
  ptr = (unsigned char *)array;
  while (length--) {
    b1 = *ptr++;
    b2 = *ptr++;
#ifdef SCI_LITTLE_ENDIAN
    *array++ = (unsigned short) ((b1 << 8) | (b2));
#else
    *array++ = (unsigned short) ((b1) | (b2 << 8));
#endif
  }
}

// Given an argument vector, remove NumDel strings from it, starting at i.
static inline void RemoveArgs(int &argc, char **argv, int &i, int NumDel=1)
{
  argc -= NumDel;
  memmove(&(argv[i]), &(argv[i+NumDel]), (argc-i) * sizeof(char *));
  i--;
}

// Note: NRand (normal distribution) is in MiscMath.h

// A random number on 0.0 to 1.0.
inline double DRand()
{
	return drand48();
}

// A random number.
inline int LRand()
{
  return lrand48();
}

// Seeds the random number generator based on time and/or process ID.
extern void SRand();

// Makes a fairly random 32-bit number from a string.
extern int HashString(const char *);


} // namespace Tools
} // namespace Remote

#endif
