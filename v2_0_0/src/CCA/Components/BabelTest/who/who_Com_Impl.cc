// 
// File:          who_Com_Impl.cc
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:27 MDT
// Generated:     20030618 13:12:33 MDT
// Description:   Server-side implementation for who.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 13
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 
#include "who_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.Com._includes)
#include "who.hh"
// DO-NOT-DELETE splicer.end(who.Com._includes)

// user defined constructor
void who::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.Com._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.Com._ctor)
}

// user defined destructor
void who::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.Com._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.Com._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Obtain Services handle, through which the 
 * component communicates with the framework. 
 * This is the one method that every CCA Component
 * must implement. 
 */
void
who::Com_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(who.Com.setServices)
  who::IDPort idPort=who::IDPort::_create();
  services.addProvidesPort(idPort,"idport","gov.cca.ports.IDPort",0);

  who::UIPort uiPort=who::UIPort::_create();
  services.addProvidesPort(uiPort,"ui","gov.cca.ports.UIPort",0);
  // DO-NOT-DELETE splicer.end(who.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(who.Com._misc)
// DO-NOT-DELETE splicer.end(who.Com._misc)

