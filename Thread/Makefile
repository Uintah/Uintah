
OBJS = AsyncReply.o Parallel.o Thread.o AtomicCounter.o \
	ParallelBase.o ThreadAlert.o Barrier.o PoolMutex.o \
	ThreadError.o ConditionVariable.o Profiler.o \
	ThreadEvent.o CrowdMonitor.o RealtimeThread.o ThreadGroup.o \
	LockUnlock.o RecursiveMutex.o ThreadListener.o Mailbox.o \
	Reducer.o ThreadTopology.o Runnable.o WorkQueue.o Mutex.o \
	Semaphore.o Thread_irix.o

CCFLAGS = -g -n32 -mips4
CCC = CC -LANG:ansi-for-init-scope=ON -MDupdate Makedepend

.SUFFIXES: .cc
.cc.o:
	$(CCC) -c $(CCFLAGS) $<

libThread.a: $(OBJS)
	$(CCC) -ar -o $@ $(CCFLAGS) $(OBJS)

include Makedepend
