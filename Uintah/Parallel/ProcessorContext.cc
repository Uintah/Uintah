/* REFERENCED */
static char *id="@(#) $Id$";

#include "ProcessorContext.h"
#include <SCICore/Thread/SimpleReducer.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Thread.h>
#include <SCICore/Exceptions/InternalError.h>
#include <iostream>

namespace Uintah {
namespace Parallel {

using SCICore::Thread::SimpleReducer;
using SCICore::Thread::Mutex;
using SCICore::Thread::Thread;
using SCICore::Exceptions::InternalError;

using std::cerr;

static Mutex lock("ProcessorContext lock");
static ProcessorContext* rootContext = 0;

ProcessorContext*
ProcessorContext::getRootContext()
{
    if(!rootContext) {
	lock.lock();
	if(!rootContext){
	    rootContext = new ProcessorContext(0,0,Thread::numProcessors(),0);
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

ProcessorContext*
ProcessorContext::createContext(int threadNumber,
				int numThreads,
				SimpleReducer* reducer) const
{
    return new ProcessorContext(this, threadNumber, numThreads, reducer);
}

void
ProcessorContext::barrier_wait() const
{
    if(!d_reducer)
	throw InternalError("ProcessorContext::reducer_wait called on a ProcessorContext that has no reducer");
    d_reducer->wait(d_numThreads);
}

double
ProcessorContext::reduce_min(double mymin) const
{
    if(!d_reducer)
	throw InternalError("ProcessorContext::reducer_wait called on a ProcessorContext that has no reducer");
    return d_reducer->min(d_threadNumber, d_numThreads, mymin);
}

} // end namespace Parallel
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
