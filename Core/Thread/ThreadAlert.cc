
#include "ThreadAlert.h"
#include <iostream.h>

/*
 * When a thread is alerted (see <b>Thread</b>), it throws
 * a <b>ThreadAlert</b> exception.  This event is typically
 * triggered asynchonously by another thread, and can therefore
 * happen at any time.  Threads should catch this exception,
 * perform any cleanup operations and rethrow the exception.
 */

int ThreadAlert::alertCode() {
    cerr << "ThreadAlert::alertCode not finished\n";
    return -1;
}

