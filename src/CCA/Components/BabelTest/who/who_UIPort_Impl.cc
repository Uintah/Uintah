// 
// File:          who_UIPort_Impl.cc
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.0
// SIDL Created:  20020813 11:08:13 CDT
// Generated:     20020813 11:08:15 CDT
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "who_UIPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
// Put additional includes or other arbitrary code here...
#include <iostream.h>
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

// user defined constructor
void who::UIPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.UIPort._ctor)
}

// user defined destructor
void who::UIPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.UIPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Execute some encapsulated functionality on the component. 
 * @return 0 if ok, -1 if internal error but component may be used further,
 * -2 if error so severe that component cannot be further used safely.
 */
int32_t
who::UIPort_impl::ui () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(who.UIPort.ui)
  cerr<<" UI button is clicked!"<<endl;
  return 0;	
  // DO-NOT-DELETE splicer.end(who.UIPort.ui)
}


// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

