#include <Datatypes/ParticleSetExtensionPort.h>

clString SimpleIPort<ParticleSetExtensionHandle>::port_type("ParticleSetExtension");
clString SimpleIPort<ParticleSetExtensionHandle>::port_color("orange");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ParticleSetExtensionHandle>;
template class SimpleOPort<ParticleSetExtensionHandle>;
template class SimplePortComm<ParticleSetExtensionHandle>;
template class Mailbox<SimplePortComm<ParticleSetExtensionHandle>*>;

#endif
