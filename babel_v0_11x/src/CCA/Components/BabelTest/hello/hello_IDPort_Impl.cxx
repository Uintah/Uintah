// 
// File:          hello_IDPort_Impl.cxx
// Symbol:        hello.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for hello.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "hello_IDPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(hello.IDPort._includes)
// Insert-Code-Here {hello.IDPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(hello.IDPort._includes)

// user defined constructor
void hello::IDPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._ctor)
  // Insert-Code-Here {hello.IDPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(hello.IDPort._ctor)
}

// user defined destructor
void hello::IDPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._dtor)
  // Insert-Code-Here {hello.IDPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(hello.IDPort._dtor)
}

// static class initializer
void hello::IDPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._load)
  // Insert-Code-Here {hello.IDPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(hello.IDPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Test prot. Return a string as an ID for Hello component
 */
::std::string
hello::IDPort_impl::getID_impl () 

{
  // DO-NOT-DELETE splicer.begin(hello.IDPort.getID)
  // What is this supposed to do???
  return ::std::string("hello::IDPort_impl::getID_Impl()");
  // DO-NOT-DELETE splicer.end(hello.IDPort.getID)
}


// DO-NOT-DELETE splicer.begin(hello.IDPort._misc)
// Insert-Code-Here {hello.IDPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(hello.IDPort._misc)

