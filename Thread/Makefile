
OBJS = AsyncReply.o Parallel.o Thread.o AtomicCounter.o \
	ParallelBase.o ThreadAlert.o Barrier.o PoolMutex.o \
	ThreadError.o ConditionVariable.o Profiler.o \
	ThreadEvent.o CrowdMonitor.o RealtimeThread.o ThreadGroup.o \
	LockUnlock.o RecursiveMutex.o ThreadListener.o Mailbox.o \
	Reducer.o ThreadTopology.o Runnable.o WorkQueue.o Mutex.o \
	Semaphore.o Thread_irix.o

TOP = ..

include ../make.config

libThread.a: $(OBJS)
	rm -f $@
	$(AR) $@ $(OBJS)

Makedepend:
	touch Makedepend

include Makedepend
