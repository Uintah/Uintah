// 
// File:          who_Com_Impl.cc
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030305 18:50:50 MST
// Generated:     20030305 18:50:56 MST
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
#include "govcca.hh"
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
 * Obtain Services handle, through which the component communicates with the
 * framework. This is the one method that every CCA Component must implement. 
 */
void
who::Com_impl::setServices (
  /*in*/ ::govcca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(who.Com.setServices)
  who::IDPort idPort=who::IDPort::_create();
  svc.addProvidesPort(idPort,"idport","gov.cca.IDPort",0);

  who::UIPort uiPort=who::UIPort::_create();
  svc.addProvidesPort(uiPort,"babel.ui","gov.cca.UIPort",0);
  // DO-NOT-DELETE splicer.end(who.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(who.Com._misc)
extern "C" govcca::Component make_Babel_who()
{
  return who::Com::_create();
}
// DO-NOT-DELETE splicer.end(who.Com._misc)

