// 
// File:          NewPort_Com_Impl.cc
// Symbol:        NewPort.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for NewPort.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "NewPort_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(NewPort.Com._includes)
#include "NewPort.hh"
// DO-NOT-DELETE splicer.end(NewPort.Com._includes)

// user-defined constructor.
void NewPort::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.Com._ctor)
  // Insert-Code-Here {NewPort.Com._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.Com._ctor)
}

// user-defined destructor.
void NewPort::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.Com._dtor)
  // Insert-Code-Here {NewPort.Com._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.Com._dtor)
}

// static class initializer.
void NewPort::Com_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.Com._load)
  // Insert-Code-Here {NewPort.Com._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.Com._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Starts up a component presence in the calling framework.
 * @param Svc the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */
void
NewPort::Com_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(NewPort.Com.setServices)
  svc = services;
  NewPort::GoPort goPort = NewPort::GoPort::_create();
  goPort.setServices(svc);
  NewPort::StringPort strPort = NewPort::StringPort::_create();

  svc.addProvidesPort(goPort, "go", "gov.cca.ports.GoPort", 0);
  svc.registerUsesPort("ustrport", "gov.cca.ports.StringPort", 0);
  svc.addProvidesPort(strPort, "pstrport", "gov.cca.ports.StringPort", 0);
  // DO-NOT-DELETE splicer.end(NewPort.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(NewPort.Com._misc)
// Insert-Code-Here {NewPort.Com._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.Com._misc)

