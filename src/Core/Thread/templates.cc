
#include <Thread/Mailbox.h>
#include <Thread/Parallel.h>
#include <Thread/TestThreads.h>

template class AsyncReply<int>;

template class Mailbox<int>;
template class Mailbox<TestThreads::TestMsg>;

template class Parallel<TestThreads>;
