
/* REFERENCED */
static char *id="$Id$";

/*
 *  TestThreads.cc: Tests for the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Thread/TestThreads.h>
#include <Thread/Guard.h>
#include <iostream.h>

TestThreads::TestThreads(RigorousTest* __test)
    : __test(__test),
      threads_seen("Number of threads that entered the tests", 0),
      mailbox0("test mailbox0", 0), mailbox2("test mailbox2", 2),
      count1("test atomic counter1", 0), count2("test atomic counter2", 0),
      barrier("test barrier", 2), monitor("test crowd monitor"),
      cond("test condition"), lock("test lock"),
      rlock("test recursive lock"), sema("test semaphore", 0),
      reducer("test reducer", 2), work("test work queue"),
      mailbox3("test mailbox3", 3)
{
}

TestThreads::~TestThreads()
{
}

void TestThreads::test1(int proc)
{
    threads_seen++;
    
    // Test Mailboxes of size 0.  Performs a simple handshake
    // If the mailbox employs the proper rendezvous semantics,
    // then this will not have a race condition.
    if(proc==0){
	for(int i=0;i<100;i++){
	    mailbox0.send(0xfacade+i);
	    TEST(mailbox0.receive() == 0xdeface+i);
	}
    } else {
	for(int i=0;i<100;i++){
	    TEST(mailbox0.receive() == 0xfacade+i);
	    mailbox0.send(0xdeface+i);
	}
    }
    TEST(mailbox0.size()==0);

    // Test Mailboxes of size 2.  This is a simple producer/consumer test.
    if(proc==0){
	for(int i=0;i<100;i++)
	    mailbox2.send(i);
    } else {
	for(int i=0;i<100;i++){
	    TEST(mailbox2.receive() == i);
	}

	// Now put items back in and make sure that they are counted
	// correctly, and that try_send works...
	TEST(mailbox2.numItems()==0);
	TEST(mailbox2.trySend(0xfeedface));
	TEST(mailbox2.numItems()==1);
	TEST(mailbox2.trySend(0xadded));
	TEST(mailbox2.numItems()==2);
	TEST(!mailbox2.trySend(0xacebabe));
	TEST(mailbox2.numItems()==2);
	
	// Now, make sure that we get them back out...
	TEST(mailbox2.receive() == 0xfeedface);
	TEST(mailbox2.numItems()==1);
	int answer;
	TEST(mailbox2.tryReceive(answer));
	TEST(answer==0xadded);
	TEST(mailbox2.numItems()==0);
	TEST(!mailbox2.tryReceive(answer));
	TEST(mailbox2.numItems()==0);
    }
    TEST(mailbox2.size()==2);
    
    // Test FutureValue...
    if(proc==0){
	for(int i=0;i<100;i++){
	    FutureValue<int> reply("test FutureValue");
	    mailbox3.send(TestMsg(0xcabfee, &reply));
	    TEST(reply == 0xcabfee+i);
	    TEST(reply == 0xcabfee+i);
	    TEST(reply == 0xcabfee+i);
	}
    } else {
	for(int i=0;i<100;i++){
	    TestMsg msg=mailbox3.receive();
	    *msg.reply=msg.data+i;
	}
    }
    
    // Test AtomicCounter...
    if(proc==0){
	for(int i=0;i<11;i++)
	    count1++;
	TEST(count1>=1);
	TEST(count1<=11);
	for(int i=0;i<11;i++)
	    ++count2;
	TEST(count2>=1);
	TEST(count2<=11);
    } else {
	for(int i=0;i<10;i++)
	    count1--;
	TEST(count1>=-10);
	TEST(count1<=1);
	for(int i=0;i<10;i++)
	    --count2;
	TEST(count2>=-10);
	TEST(count2<=1);
    }
    
    // Test barrier...
    for(int i=0;i<1000;i++){
	barrier.wait();
	if(proc==0)
	    val=i;
	barrier.wait();
	TEST(val==i);
	barrier.wait();
	if(proc==1)
	    val=-i;
	barrier.wait();
	TEST(val==-i);
    }
    
    barrier.wait();
    // Test ConditionVariable...
#if 0
    if(proc==0){
	lock.lock();
	barrier.wait();
	cond.wait(lock);
	TEST(val==0xfeeddad);
	val=0xefface;
	cond.conditionBroadcast();
	lock.unlock();
    } else {
	val=0xfeeddad;
	sema.down();
	lock.lock();
	cond.conditionSignal();
	cond.wait(lock);
	lock.unlock();
	TEST(val==0xefface);
    }
#endif
    
    barrier.wait();
    val=0;
    barrier.wait();
    
    // Test CrowdMonitor...
    for(int i=0;i<100;i++){
	monitor.writeLock();
	val++;
	TEST(val>=i+1);
	TEST(val<=i+11);
	monitor.writeUnlock();
	monitor.readLock();
	int oval=val;
	for(int ii=0;ii<1000;ii++){
	    TEST(val == oval);
	}
	monitor.readUnlock();
    }
    TEST(val>=10);
    TEST(val<=20);
    barrier.wait();
    TEST(val==20);
    
    barrier.wait();
    val=0;
    barrier.wait();
    // Test Guard...
    for(int i=0;i<1000;i++){
	Guard t(&lock);
	val++;
	TEST(val>=i+1);
	TEST(val<=i+1001);
    }
    TEST(val>=1000);
    TEST(val<=2000);
    barrier.wait();
    TEST(val==2000);
    barrier.wait();
    val=0;
    barrier.wait();
    for(int i=0;i<100;i++){
	{
	    Guard t(&monitor, Guard::Write);
	    val++;
	    TEST(val>=i+1);
	    TEST(val<=i+11);
	}
	{
	    Guard t(&monitor, Guard::Read);
	    int oval=val;
	    for(int ii=0;ii<1000;ii++){
		TEST(val == oval);
	    }
	}
    }
    TEST(val>=100);
    TEST(val<=200);
    barrier.wait();
    TEST(val==200);
    
    barrier.wait();
    val=0;
    barrier.wait();
    // Test Mutex...
    for(int i=0;i<1000;i++){
	lock.lock();
	val++;
	TEST(val>=i+1);
	TEST(val<=i+1001);
	lock.unlock();
    }
    TEST(val>=1000);
    TEST(val<=2000);
    barrier.wait();
    TEST(val==2000);
    
    // More mutex tests...
    if(proc==0){
	for(int i=0;i<100;i++){
	    barrier.wait();
	    TEST(!lock.tryLock()); // B
	    barrier.wait();
	    barrier.wait();
	    TEST(lock.tryLock()); // D
	    lock.unlock();
	    barrier.wait();
	}
    } else {
	for(int i=0;i<100;i++){
	    lock.lock(); // A
	    barrier.wait();
	    barrier.wait();
	    lock.unlock(); // C
	    barrier.wait();
	    barrier.wait();
	}
    }
    val=0;
    barrier.wait();
#if 0
    // Pool mutex tests...
    for(int i=0;i<100;i++){
	plock.lock();
	val++;
	TEST(val>=i+1);
	TEST(val<=i+101);
	plock.unlock();
    }
    TEST(val>=100);
    TEST(val<=200);
    barrier.wait();
    TEST(val==200);
    if(proc==0){
	sema.down();
	TEST(!plock.tryLock());
	sema.down();
	TEST(plock.tryLock());
	plock.unlock();
    } else {
	plock.lock();
	sema.up();
	plock.unlock();
	sema.up();
    }
#endif
    barrier.wait();
    val=0;
    barrier.wait();
    
    // Recursive mutex tests...
    for(int i=0;i<100;i++){
	rlock.lock();
	val++;
	TEST(val>=3*i+1);
	TEST(val<=3*i+301);

	rlock.lock();
	val++;
	TEST(val>=3*i+2);
	TEST(val<=3*i+302);
	rlock.unlock();
	
	rlock.lock();
	val++;
	TEST(val>=3*i+3);
	TEST(val<=3*i+303);
	rlock.unlock();
	
	rlock.unlock();
    }
    TEST(val>=300);
    TEST(val<=600);
    barrier.wait();
    TEST(val==600);
    
    // Test Reducer
    for(int i=0;i<1000;i++){
	double s1=reducer.sum(proc, i+proc);
	TEST(s1==2*i+1);
	double s2=reducer.max(proc, ((i%2)*2-1)*(proc*2-1));
	TEST(s2==1);
    }
    // Test Semaphore
    if(proc==0){
	for(int i=0;i<100;i++){
	    sema.up();
	}
    } else {
	for(int i=0;i<100;i++){
	    sema.up();
	}	    
	for(int i=0;i<200;i++){
	    sema.down();
	}
    }
    barrier.wait();
    if(proc==0){
	sema.up();
	TEST(sema.tryDown());
	TEST(!sema.tryDown());
    }
    // Test WorkQueue
    for(int i=0;i<50;i++){
	if(proc==0)
	    work.refill(200, 2, false);
	barrier.wait();
	int start, end;
	int sum=0;
	while(work.nextAssignment(start, end)){
	    TEST(end>start);
	    TEST(start>=0);
	    TEST(end<=200);
	    for(int i=start;i<end;i++){
		sum+=i;
		for(int j=0;j<(proc+1)*100;j++)
		    val+=j;
	    }
	}
	wsum[proc]=sum;
	barrier.wait();
	TEST(wsum[0]+wsum[1]==(200*200-200)/2);
    }
#if 0
    for(int i=0;i<5;i++){
	barrier.wait();
	if(proc==0)
	    work.refill(20, 2, true);
	barrier.wait();
	int start, end;
	int sum=0;
	wsum[proc]=0;
	while(work.nextAssignment(start, end)){
	    TEST(end>start);
	    TEST(start>=0);
	    TEST(end<=40);
	    for(int i=start;i<end;i++){
		sum+=i;
		if(i==19)
		    work.addWork(20);
		for(int j=0;j<proc*100;j++)
		    val+=j;
		}	
	    wsum[proc]=sum;
	}
	TEST(wsum[0]+wsum[1]==(40*40-40)/2);
    }
#endif

    // Test Thread internals (just the ones that aren't already used...)
    Thread* me=Thread::currentThread();
    TEST(!me->isDaemon());
    g=me->threadGroup();
    TEST(g != 0);
    barrier.wait();
    TEST(me->threadGroup() == g);

    for(int i=1;i<10;i++){
	me->setPriority(i);
	for(int j=0;j<proc*100;j++)
	    val+=j;
	TEST(me->getPriority() == i);
    }
    me->setPriority(5);
    if(proc==0)
	t0=me;
    barrier.wait();
#if 0
    if(proc==1){
	t0->stop();
	t0->resume();
    }
#endif

}
//
// $Log$
// Revision 1.1  1999/08/25 02:38:09  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

