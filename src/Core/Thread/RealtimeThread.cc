
#include "RealtimeThread.h"
#include "Thread.h"

/*
 * Extended threads with realtime capabilities.  These realtime
 * capabilities may not be available on all machines, but the
 * functionality will be approximated on those that do not.
 *
 * <p>On multiprocessor machines, the realtime threads typically
 * take control of a single CPU.  On uniprocessor machines, they
 * can consume a large percentage of the processing resources
 * if the interval is too small.
 *
 * <p>Until this thread calls <i>frameSchedule</i>, this thread
 * behaves exactly like any other thread.
 */

int RealtimeThread::frameInterval() {
    return interval;
}

