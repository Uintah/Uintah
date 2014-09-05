// 
// File:          NewPort_GoPort_Impl.cxx
// Symbol:        NewPort.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for NewPort.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "NewPort_GoPort_Impl.hxx"

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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(NewPort.GoPort._includes)
#include "NewPort.hxx"
#include <iostream>
// DO-NOT-DELETE splicer.end(NewPort.GoPort._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
NewPort::GoPort_impl::GoPort_impl() : StubBase(reinterpret_cast< void*>(
  ::NewPort::GoPort::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._ctor2)
  // Insert-Code-Here {NewPort.GoPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._ctor2)
}

// user defined constructor
void NewPort::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._ctor)
  // Insert-Code-Here {NewPort.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._ctor)
}

// user defined destructor
void NewPort::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._dtor)
  // Insert-Code-Here {NewPort.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._dtor)
}

// static class initializer
void NewPort::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._load)
  // Insert-Code-Here {NewPort.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setServices[]
 */
void
NewPort::GoPort_impl::setServices_impl (
  /* in */::gov::cca::Services& svc ) 
{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.setServices)
  // Insert-Code-Here {NewPort.GoPort.setServices} (setServices method)
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
NewPort::GoPort_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.go)
  // Insert-Code-Here {NewPort.GoPort.go} (go method)
  ::gov::cca::Port p = svc.getPort("ustrport");
  ::NewPort::StringPort s = ::sidl::babel_cast< ::NewPort::StringPort>(p);
  if (s._is_nil()) {
    std::cerr << "getPort() returns null" << std::endl;
    return -1;
  }
  std::cerr << "Received " << s.getString() << std::endl;
  return 0;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(NewPort.GoPort._misc)
// Insert-Code-Here {NewPort.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.GoPort._misc)

