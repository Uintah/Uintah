#include <Datatypes/ParticleGridReaderPort.h>

clString SimpleIPort<ParticleGridReaderHandle>::port_type("ParticleGridReader");
clString SimpleIPort<ParticleGridReaderHandle>::port_color("cyan");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ParticleGridReaderHandle>;
template class SimpleOPort<ParticleGridReaderHandle>;
template class SimplePortComm<ParticleGridReaderHandle>;
template class Mailbox<SimplePortComm<ParticleGridReaderHandle>*>;

#endif
