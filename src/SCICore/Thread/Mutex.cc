
#include "Mutex.h"

/*
 * Provides a simple <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
 * <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
 * This is not a recursive Mutex (See <b>RecursiveMutex</b>), and calling
 * lock() in a nested call will result in an error or deadlock.
 */

