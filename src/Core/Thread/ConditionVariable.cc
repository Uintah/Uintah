
#include "ConditionVariable.h"
#include "Thread.h"

/**
 * Condition variable primitive.  When a thread calls the
 * <i>wait</i> method,which will block until another thread calls
 * the <i>cond_signal</i> or <i>cond_broadcast</i> methods.  When
 * there are multiple threads waiting, <i>cond_broadcast</i> will unblock
 * all of them, while <i>cond_signal</i> will unblock only one (an
 * arbitrary one) of them.  This primitive is used to allow a thread
 * to wait for some condition to exist, such as an available resource.
 * The thread waits for that condition, until it is unblocked by another
 * thread that caused the condition to exist (<i>i.e.</i> freed the
 * resource).
 */

ConditionVariable::ConditionVariable(const char* name)
	: name(name), nwaiters(0), mutex("Condition variable lock"),
      semaphore("Condition variable semaphore", 0)
 {
 }

ConditionVariable::~ConditionVariable() {
}

void ConditionVariable::wait(Mutex& m) {
    mutex.lock();
    nwaiters++;
    mutex.unlock();
    m.unlock();
    // Block until woken up by signal or broadcast
    int s=Thread::couldBlock(name);
    semaphore.down();
    Thread::couldBlock(s);
    m.lock();
}

void ConditionVariable::cond_signal()  {
    mutex.lock();
    if(nwaiters > 0){
        nwaiters--;
        semaphore.up();
    }
    mutex.unlock();
}

void ConditionVariable::cond_broadcast() {
    mutex.lock();
    while(nwaiters > 0){
        nwaiters--;
        semaphore.up();
    }
    mutex.unlock();
}

