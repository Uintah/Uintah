
/*
 * Assists in the testing of threads.
 */

#include <Tester/RigorousTest.h>
#include <Thread/AtomicCounter.h>
#include <Thread/Barrier.h>
#include <Thread/ConditionVariable.h>
#include <Thread/CrowdMonitor.h>
#include <Thread/FutureValue.h>
#include <Thread/Mailbox.h>
#include <Thread/Mutex.h>
#include <Thread/RecursiveMutex.h>
#include <Thread/Reducer.h>
#include <Thread/Semaphore.h>
#include <Thread/WorkQueue.h>

class TestThreads {
    RigorousTest* __test;
public:
    TestThreads(RigorousTest* __test);
    ~TestThreads();

    AtomicCounter threads_seen;
    Mailbox<int> mailbox0;
    Mailbox<int> mailbox2;
    AtomicCounter count1;
    AtomicCounter count2;
    volatile int val;
    Barrier barrier;
    CrowdMonitor monitor;
    ConditionVariable cond;
    Mutex lock;
    RecursiveMutex rlock;
    Semaphore sema;
    Reducer reducer;
    WorkQueue work;
    int wsum[2];
    ThreadGroup* g;
    Thread* t0;

    struct TestMsg {
	int data;
	FutureValue<int>* reply;
	TestMsg(int data, FutureValue<int>* reply) : data(data), reply(reply)
	    {}
	TestMsg() {}
    };
    Mailbox<TestMsg> mailbox3;

    /*
     * Perform some short tests with two threads...
     */
    void test1(int proc);
};
