
#include "AtomicCounter.h"

/*
 * Provides a simple atomic counter.  This will work just like an
 * integer, but guarantees atomicty of the ++ and -- operators.
 * Despite their convenience, you do not want to make a large number
 * of these objects.  See also <b>WorkQueue</b>.
 */

AtomicCounter::AtomicCounter(const char* name) : name(name), lock("AtomicCounter lock") {
}

AtomicCounter::AtomicCounter(const char* name, int value) : name(name), lock("AtomicCounter lock"), value(value) {
}

AtomicCounter::~AtomicCounter() {
}

AtomicCounter::operator int() const {
    return value;
}

AtomicCounter& AtomicCounter::operator++() {
    lock.lock();
    ++value;
    lock.unlock();
    return *this;
}

int AtomicCounter::operator++(int) {
    lock.lock();
    int ret=value++;
    lock.unlock();
    return ret;
}

AtomicCounter& AtomicCounter::operator--() {
    lock.lock();
    --value;	
    lock.unlock();
    return *this;
}

int AtomicCounter::operator--(int) {
    lock.lock();
    int ret=value--;
    lock.unlock();
    return ret;
}

