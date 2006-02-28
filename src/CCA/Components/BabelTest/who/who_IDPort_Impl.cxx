// 
// File:          who_IDPort_Impl.cxx
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "who_IDPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Insert-Code-Here {who.IDPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

// user defined constructor
void who::IDPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._ctor)
  // Insert-Code-Here {who.IDPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(who.IDPort._ctor)
}

// user defined destructor
void who::IDPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._dtor)
  // Insert-Code-Here {who.IDPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(who.IDPort._dtor)
}

// static class initializer
void who::IDPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._load)
  // Insert-Code-Here {who.IDPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(who.IDPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Test prot. Return a string as an ID for Hello component
 */
::std::string
who::IDPort_impl::getID_impl () 

{
  // DO-NOT-DELETE splicer.begin(who.IDPort.getID)
  return ::std::string("World (in C++)");
  // DO-NOT-DELETE splicer.end(who.IDPort.getID)
}


// DO-NOT-DELETE splicer.begin(who.IDPort._misc)
// Insert-Code-Here {who.IDPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(who.IDPort._misc)

