
#include "ConditionVariable.h"
#include "Thread.h"

/*
 * Condition variable primitive.  When a thread calls the <i>wait</i>
 * method,which will block until another thread calls the
 * <i>conditionSignal</i> or <i>conditionBroadcast</i> methods.  When
 * there are multiple threads waiting, <i>conditionBroadcast</i> will
 * unblock all of them, while <i>conditionSignal</i> will unblock only
 * one (an arbitrary one) of them.  This primitive is used to allow a
 * thread to wait for some condition to exist, such as an available
 * resource.  The thread waits for that condition, until it is
 * unblocked by another thread that caused the condition to exist
 * (<i>i.e.</i> freed the resource).
*/

ConditionVariable::ConditionVariable(const std::string& name)
    : d_name(name), d_numWaiters(0), d_mutex("Condition variable lock"),
      d_semaphore("Condition variable semaphore", 0)
{
}

ConditionVariable::~ConditionVariable()
{
}

void ConditionVariable::wait(Mutex& m)
{
    d_mutex.lock();
    d_numWaiters++;
    d_mutex.unlock();
    m.unlock();
    // Block until woken up by signal or broadcast
    int s=Thread::couldBlock(d_name);
    d_semaphore.down();
    Thread::couldBlockDone(s);
    m.lock();
}

void ConditionVariable::conditionSignal()
{
    d_mutex.lock();
    if(d_numWaiters > 0){
        d_numWaiters--;
        d_semaphore.up();
    }
    d_mutex.unlock();
}

void ConditionVariable::conditionBroadcast()
{
    d_mutex.lock();
    while(d_numWaiters > 0){
        d_numWaiters--;
        d_semaphore.up();
    }
    d_mutex.unlock();
}

