// 
// File:          NewPort_GoPort_Impl.cc
// Symbol:        NewPort.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040301 18:37:59 MST
// Generated:     20040301 18:38:04 MST
// Description:   Server-side implementation for NewPort.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 6
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/NewPort/NewPort.sidl
// 
#include "NewPort_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(NewPort.GoPort._includes)
#include "NewPort.hh"
#include <iostream>
// DO-NOT-DELETE splicer.end(NewPort.GoPort._includes)

// user defined constructor
void NewPort::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._ctor)
}

// user defined destructor
void NewPort::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setService[]
 */
void
NewPort::GoPort_impl::setService (
  /*in*/ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.setService)
  this->svc=svc;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.setService)
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
  NewPort::StringPort s=svc.getPort("ustrport");
  std::cerr << "Got the port\n";
  if(!s._is_nil()) std::cerr<<"Received "<< s.getString() <<"\n";
  else std::cerr<<"getPort() returns null\n";
  return 0;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(NewPort.GoPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(NewPort.GoPort._misc)

