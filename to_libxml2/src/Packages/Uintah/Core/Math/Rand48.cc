#if defined(_WIN32) && !defined(HAVE_DRAND48)
#include <Core/Thread/Time.h>
#include <stdlib.h>
static bool initialized = false;

static void drand_initialize()
{
  initialized = true;
}

double drand48()
{
  if (!initialized) drand_initialize();
  return ((double) rand())/RAND_MAX;
}
#endif
