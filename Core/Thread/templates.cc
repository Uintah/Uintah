
#include <SCICore/Thread/Mailbox.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/TestThreads.h>

template class AsyncReply<int>;

template class Mailbox<int>;
template class Mailbox<TestThreads::TestMsg>;

template class Parallel<TestThreads>;
