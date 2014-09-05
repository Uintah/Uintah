// 
// File:          who_IDPort_Impl.cc
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:24 MDT
// Generated:     20030618 13:12:33 MDT
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 
#include "who_IDPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

// user defined constructor
void who::IDPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.IDPort._ctor)
}

// user defined destructor
void who::IDPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.IDPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Test prot. Return a string as an ID for Hello component
 */
::std::string
who::IDPort_impl::getID () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(who.IDPort.getID)
  return "World (in C++)";
  // DO-NOT-DELETE splicer.end(who.IDPort.getID)
}


// DO-NOT-DELETE splicer.begin(who.IDPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(who.IDPort._misc)

