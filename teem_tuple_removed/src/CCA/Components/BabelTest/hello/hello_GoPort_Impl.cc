// 
// File:          hello_GoPort_Impl.cc
// Symbol:        hello.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:33 MDT
// Generated:     20030618 13:12:40 MDT
// Description:   Server-side implementation for hello.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/hello/hello.sidl
// 
#include "hello_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(hello.GoPort._includes)
#include "gov_cca_ports_GoPort.hh"
#include "gov_cca_ports_IDPort.hh"
#include <iostream.h>
// DO-NOT-DELETE splicer.end(hello.GoPort._includes)

// user defined constructor
void hello::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(hello.GoPort._ctor)
}

// user defined destructor
void hello::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(hello.GoPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setService[]
 */
void
hello::GoPort_impl::setService (
  /*in*/ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.setService)
  this->svc=svc;
  // DO-NOT-DELETE splicer.end(hello.GoPort.setService)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
hello::GoPort_impl::go () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.go)
  cerr<<"hello::GoPort::go() starts..."<<endl;
  gov::cca::ports::IDPort s=svc.getPort("idport");
  if(!s._is_nil())   cerr<<"Hello "<<s.getID()<<endl;
  else cerr<<"getPort() returns null"<<endl;
  // DO-NOT-DELETE splicer.end(hello.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(hello.GoPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(hello.GoPort._misc)

