
/*
 * Assists in the testing of threads.
 */

#include <SCICore/Tester/RigorousTest.h>
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Barrier.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <SCICore/Thread/FutureValue.h>
#include <SCICore/Thread/Mailbox.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/RecursiveMutex.h>
#include <SCICore/Thread/Reducer.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/WorkQueue.h>

namespace SCICore {
    namespace Thread {
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
    }
}

//
// $Log$
// Revision 1.1  1999/08/25 02:38:09  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

