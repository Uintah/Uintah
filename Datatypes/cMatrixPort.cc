#include <Datatypes/cMatrixPort.h>

clString SimpleIPort<cMatrixHandle>::port_type("cMatrix");
clString SimpleIPort<cMatrixHandle>::port_color("red");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<cMatrixHandle>;
template class SimpleOPort<cMatrixHandle>;
template class SimplePortComm<cMatrixHandle>;
template class Mailbox<SimplePortComm<cMatrixHandle>*>;

#endif
