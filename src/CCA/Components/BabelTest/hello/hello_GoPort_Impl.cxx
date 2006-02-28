// 
// File:          hello_GoPort_Impl.cxx
// Symbol:        hello.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for hello.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "hello_GoPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(hello.GoPort._includes)
//#include "gov_cca_ports_GoPort.hh"
#include "gov_cca_ports_IDPort.hxx"
#include <iostream.h>
// DO-NOT-DELETE splicer.end(hello.GoPort._includes)

// user defined constructor
void hello::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._ctor)
  // Insert-Code-Here {hello.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(hello.GoPort._ctor)
}

// user defined destructor
void hello::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._dtor)
  // Insert-Code-Here {hello.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(hello.GoPort._dtor)
}

// static class initializer
void hello::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._load)
  // Insert-Code-Here {hello.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(hello.GoPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setServices[]
 */
void
hello::GoPort_impl::setServices_impl (
  /* in */UCXX ::gov::cca::Services services ) 
{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.setServices)
  this->svc = services;
  // DO-NOT-DELETE splicer.end(hello.GoPort.setServices)
}

/**
 * Execute some encapsulated functionality on the component.
 * Return 0 if ok, -1 if internal error but component may be
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
hello::GoPort_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.go)
  ::std::cerr << "hello::GoPort::go() starts..." << ::std::endl;
  UCXX ::gov::cca::Port p = svc.getPort("idport");
  UCXX ::gov::cca::ports::IDPort s = UCXX ::sidl::babel_cast<UCXX ::gov::cca::ports::IDPort>(p);
  if (s._not_nil()) {
    ::std::cerr << "Hello " << s.getID() << ::std::endl;
  } else {
    ::std::cerr << "getPort() returns null" << ::std::endl;
  }
  // DO-NOT-DELETE splicer.end(hello.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(hello.GoPort._misc)
// Insert-Code-Here {hello.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(hello.GoPort._misc)

