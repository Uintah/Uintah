
#include "PoolMutex.h"
#include "Mutex.h"

/*
 * Provides a simple mutual exclusion primitive.  This
 * differs from the <b>Mutex</b> class in that the mutexes are allocated
 * from a static pool of mutexes.  A single mutex may get assigned to
 * more than one object.  This will still provide the atomicity
 * guaranteed by a mutex, but requires significantly less memory.  Since
 * the mutex may be associated with many other objects, <b>PoolMutex</b>
 * should only be used in low-use, short duration scenarios.  As with
 * <b>Mutex</b>, <b>lock()</b> and <b>unlock()</b> will lock and unlock
 * the mutex, and <b>PoolMutex</b> is not a recursive Mutex (See
 * <b>RecursiveMutex</b>), and calling lock() in a nested call will
 * result in an error or deadlock.
 */

