//////////////////////////////////////////////////////////////////////
// Utils.cpp - Various utility functions.
//
// By Dave McAllister, 1998.

#include <Remote/Modules/remoteSalmon/toolconfig.h>
#include <Remote/Modules/remoteSalmon/Utils.h>

#ifdef SCI_MACHINE_win
#include <time.h>
#endif

#ifdef SCI_MACHINE_sgi
#include <time.h>
#include <unistd.h>
#endif

#ifdef SCI_MACHINE_hp
#include <time.h>
#include <unistd.h>
#endif

#ifdef SCI_MACHINE_win
void SRand()
{
	srand(time(0));
}
#endif

#ifdef SCI_MACHINE_sgi
void SRand()
{
	srand48(time(0) * getpid());
}
#endif

#ifdef SCI_MACHINE_hp
void SRand()
{
	srand48(time(0) * getpid());
}
#endif

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
