
#include "RecursiveMutex.h"
#include "Thread.h"

/*
 * Provides a recursive <b>Mut</b>ual <b>Ex</b>clusion primitive.  Atomic
 * <b>lock()</b> and <b>unlock()</b> will lock and unlock the mutex.
 * Nested calls to <b>lock()</b> by the same thread are acceptable,
 * but must be matched with calls to <b>unlock()</b>.  This class
 * may be less efficient that the <b>Mutex</b> class, and should not
 * be used unless the recursive lock feature is really required.
 */

RecursiveMutex::RecursiveMutex(const char* name) : mylock(name) {
    owner=0;
    lock_count=0;
}

RecursiveMutex::~RecursiveMutex() {
}

void RecursiveMutex::lock() {
    Thread* me=Thread::currentThread();
    if(owner == me){
        lock_count++;
        return;
    }
    mylock.lock();
    owner=me;
    lock_count=1;
}

void RecursiveMutex::unlock() {
    if(--lock_count == 0){
        owner=0;
        mylock.unlock();
    }
}

