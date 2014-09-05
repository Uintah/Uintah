// 
// File:          framework_Port_Impl.cxx
// Symbol:        framework.Port-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for framework.Port
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "framework_Port_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(framework.Port._includes)
// Insert-Code-Here {framework.Port._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(framework.Port._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
framework::Port_impl::Port_impl() : StubBase(reinterpret_cast< void*>(
  ::framework::Port::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(framework.Port._ctor2)
  // insert code here (ctor2)
  // DO-NOT-DELETE splicer.end(framework.Port._ctor2)
}

// user defined constructor
void framework::Port_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Port._ctor)
  // Insert-Code-Here {framework.Port._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(framework.Port._ctor)
}

// user defined destructor
void framework::Port_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Port._dtor)
  // Insert-Code-Here {framework.Port._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(framework.Port._dtor)
}

// static class initializer
void framework::Port_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.Port._load)
  // Insert-Code-Here {framework.Port._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.Port._load)
}

// user defined static methods: (none)

// user defined non-static methods: (none)

// DO-NOT-DELETE splicer.begin(framework.Port._misc)
// Insert-Code-Here {framework.Port._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(framework.Port._misc)

