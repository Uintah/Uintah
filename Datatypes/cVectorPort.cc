
#include <Datatypes/cVectorPort.h>

clString SimpleIPort<cVectorHandle>::port_type("cVector");
clString SimpleIPort<cVectorHandle>::port_color("yellow");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<cVectorHandle>;
template class SimpleOPort<cVectorHandle>;
template class SimplePortComm<cVectorHandle>;
template class Mailbox<SimplePortComm<cVectorHandle>*>;

#endif
