
#include "Semaphore.h"

/*
 * Counting semaphore synchronization primitive.  A semaphore provides
 * atomic access to a special counter.  The <i>up</i> method is used
 * to increment the counter, and the <i>down</i> method is used to
 * decrement the counter.  If a thread tries to decrement the counter
 * when the counter is zero, that thread will be blocked until another
 * thread calls the <i>up</i> method.
 */

