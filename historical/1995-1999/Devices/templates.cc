
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Multitask/Mailbox.cc>

class MessageBase;
template class Mailbox<MessageBase*>;
class Tracker;
template class Array1<Tracker*>;

