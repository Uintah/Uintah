// 
// File:          NewPort_Com_Impl.cc
// Symbol:        NewPort.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040301 18:38:02 MST
// Generated:     20040301 18:38:04 MST
// Description:   Server-side implementation for NewPort.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 18
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/NewPort/NewPort.sidl
// 
#include "NewPort_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(NewPort.Com._includes)
#include "NewPort.hh"
// DO-NOT-DELETE splicer.end(NewPort.Com._includes)

// user defined constructor
void NewPort::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.Com._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(NewPort.Com._ctor)
}

// user defined destructor
void NewPort::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.Com._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(NewPort.Com._dtor)
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
NewPort::Com_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(NewPort.Com.setServices)
  svc=services;
  NewPort::GoPort goPort=NewPort::GoPort::_create();
  goPort.setService(svc);
  NewPort::StringPort strPort=NewPort::StringPort::_create();

  svc.addProvidesPort(goPort,"go","gov.cca.ports.GoPort",0);
  svc.registerUsesPort("ustrport","gov.cca.ports.StringPort",0);
  svc.addProvidesPort(strPort,"pstrport","gov.cca.ports.StringPort",0);
  // DO-NOT-DELETE splicer.end(NewPort.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(NewPort.Com._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(NewPort.Com._misc)

