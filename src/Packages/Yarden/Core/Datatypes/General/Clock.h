
#ifndef FAST_CLOCK_H
#define FAST_CLOCK_H

namespace Yarden {
  namespace Datatypes {

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

#define CYCLE_COUNTER_IS_64BIT

#ifdef CYCLE_COUNTER_IS_64BIT
typedef unsigned long long iotimer_t;
#else
typedef unsigned int iotimer_t;
#endif

extern volatile iotimer_t counter_value, *iotimer_addr;
extern unsigned int cycleval;

inline iotimer_t read_time(void) { return *(iotimer_addr); }
void init_clock();
void PrintTime(iotimer_t s, iotimer_t e, char *txt);

  }
}

#endif
