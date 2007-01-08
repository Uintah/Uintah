#ifndef CORE_OS_RAND_H
#define CORE_OS_RAND_H

#include <Core/Thread/Time.h>

using SCIRun::Time;

inline double drand48()
{
  static bool initialized = false;
  if (!initialized) {
    srand((int) Time::currentTicks());
    initialized = true;
  }
  return ((double) rand())/ RAND_MAX;
}


#endif