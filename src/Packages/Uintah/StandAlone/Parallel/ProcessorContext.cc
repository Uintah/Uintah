
#include "ProcessorContext.h"
#include <SCICore/Thread/SimpleReducer.h>
using SCICore::Thread::SimpleReducer;
#include <SCICore/Thread/Mutex.h>
using SCICore::Thread::Mutex;
#include <SCICore/Thread/Thread.h>
using SCICore::Thread::Thread;
#include <SCICore/Exceptions/InternalError.h>
using SCICore::Exceptions::InternalError;
#include <iostream>
using std::cerr;

static Mutex lock("ProcessorContext lock");
static ProcessorContext* rootContext = 0;

ProcessorContext* ProcessorContext::getRootContext()
{
    if(!rootContext) {
	lock.lock();
	if(!rootContext){
	    rootContext = new ProcessorContext(0, 0, Thread::numProcessors(), 0);
	}
	lock.unlock();
    }
    return rootContext;
}

ProcessorContext::ProcessorContext(const ProcessorContext* parent,
				   int threadNumber, int numThreads,
				   SimpleReducer* reducer)
    : d_parent(parent), d_threadNumber(threadNumber), d_numThreads(numThreads),
      d_reducer(reducer)
{
}

ProcessorContext::~ProcessorContext()
{
}

ProcessorContext* ProcessorContext::createContext(int threadNumber,
						  int numThreads,
						  SimpleReducer* reducer) const
{
    return new ProcessorContext(this, threadNumber, numThreads, reducer);
}

void ProcessorContext::barrier_wait() const
{
    if(!d_reducer)
	throw InternalError("ProcessorContext::reducer_wait called on a ProcessorContext that has no reducer");
    d_reducer->wait(d_numThreads);
}

double ProcessorContext::reduce_min(double mymin) const
{
    if(!d_reducer)
	throw InternalError("ProcessorContext::reducer_wait called on a ProcessorContext that has no reducer");
    return d_reducer->min(d_threadNumber, d_numThreads, mymin);
}

