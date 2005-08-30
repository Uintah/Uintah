// 
// File:          NewPort_GoPort_Impl.cc
// Symbol:        NewPort.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for NewPort.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "NewPort_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(NewPort.GoPort._includes)
#include "NewPort.hh"
#include <iostream>
// DO-NOT-DELETE splicer.end(NewPort.GoPort._includes)

// user-defined constructor.
void NewPort::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._ctor)
  // Insert-Code-Here {NewPort.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._ctor)
}

// user-defined destructor.
void NewPort::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._dtor)
  // Insert-Code-Here {NewPort.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._dtor)
}

// static class initializer.
void NewPort::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._load)
  // Insert-Code-Here {NewPort.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setServices[]
 */
void
NewPort::GoPort_impl::setServices (
  /* in */ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.setServices)
  this->svc = svc;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.setServices)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
NewPort::GoPort_impl::go ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.go)
  NewPort::StringPort s = svc.getPort("ustrport");
  std::cerr << "Got the port\n";
  if (!s._is_nil()) {
    std::cerr<<"Received "<< s.getString() <<"\n";
  } else {
    std::cerr<<"getPort() returns null\n";
  }
  return 0;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(NewPort.GoPort._misc)
// Insert-Code-Here {NewPort.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.GoPort._misc)

