//static char *id="@(#) $Id$";

/*
  Clock.h
  

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef FAST_CLOCK_H
#define FAST_CLOCK_H

namespace Phil {
namespace Modules {

typedef unsigned int TimerVal;
struct ProfTimes {
  float time;
  int   numNodes;
};


// this is global data for doing the timing stuff...

#include <stddef.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syssgi.h>
#include <stdio.h>

#define CYCLE_COUNTER_IS_64BIT

#ifdef CYCLE_COUNTER_IS_64BIT
typedef unsigned long long iotimer_t;
#else
typedef unsigned int iotimer_t;
#endif

extern volatile iotimer_t counter_value, *iotimer_addr;
extern unsigned int cycleval;

inline iotimer_t read_time(void) { return *(iotimer_addr); }
void init_clock(int n = 1);
void fPrintTime(FILE *fp, iotimer_t s, iotimer_t e, char *txt);
void PrintTime(iotimer_t s, iotimer_t e, char *txt);
void AddTime(iotimer_t s, iotimer_t e, int id);
void PrintAvgTime(FILE *fp, int id, int n, char *txt);
void ClearTime( int id );

static double *clocks;

} // end namespace Modules
} // end namespace Phil

#endif

//
// $Log$
// Revision 1.2  2000/02/04 22:11:13  psutton
// fixed ID problem
//
// Revision 1.1  2000/02/04 20:56:52  psutton
// initial revision
//
//
