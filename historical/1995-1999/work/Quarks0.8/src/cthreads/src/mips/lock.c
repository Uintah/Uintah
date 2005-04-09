/*
 * This code makes no pretense of support for multiprocessors.
 * We simply disable the timer interupts for this processor and
 * assume that we therefore have exclusive access to the lock 
 * variable.
 */
#include <assert.h>
#include "lock.h"

int 
spin_try_lock(spin_lock_t *s)
{
  int result;

  splhigh();
  result = (*s == 0);
  if (result)  *s = 1;
  spllow();

  return result;
}

