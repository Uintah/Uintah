//////////////////////////////////////////////////////////////////////
// Utils.cpp - Various utility functions.
// By Dave McAllister, 1998.

#include <Packages/Remote/Tools/Util/Utils.h>

#include <time.h>
#include <unistd.h>

namespace Remote {
void SRand()
{
	srand48(time(0) * getpid());
}

// Makes a fairly random 32-bit number from a string.
int HashString(const char *s)
{
  int H = 0, i = 0, j = 0;
  while(*s)
    {
      H ^= int(*s) << i;
      i += 6;

      if(i > 24)
	{
	  j = (j+1) % 8;
	  i = j;
	}
      s++;
    }

  return H;
}
} // End namespace Remote

