// 
// File:          who_UIPort_Impl.cxx
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "who_UIPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
#include <iostream.h>
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

// user defined constructor
void who::UIPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._ctor)
  // Insert-Code-Here {who.UIPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(who.UIPort._ctor)
}

// user defined destructor
void who::UIPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._dtor)
  // Insert-Code-Here {who.UIPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(who.UIPort._dtor)
}

// static class initializer
void who::UIPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._load)
  // Insert-Code-Here {who.UIPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(who.UIPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  ui[]
 */
int32_t
who::UIPort_impl::ui_impl () 

{
  // DO-NOT-DELETE splicer.begin(who.UIPort.ui)
  ::std::cerr << " UI button is clicked!" << ::std::endl;
  return 0;	
  // DO-NOT-DELETE splicer.end(who.UIPort.ui)
}


// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Insert-Code-Here {who.UIPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

