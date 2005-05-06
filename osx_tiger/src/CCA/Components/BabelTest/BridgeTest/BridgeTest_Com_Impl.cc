// 
// File:          BridgeTest_Com_Impl.cc
// Symbol:        BridgeTest.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040412 13:28:09 MST
// Generated:     20040412 13:28:12 MST
// Description:   Server-side implementation for BridgeTest.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 18
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/BridgeTest/BridgeTest.sidl
// 
#include "BridgeTest_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(BridgeTest.Com._includes)
#include "BridgeTest.hh"
// DO-NOT-DELETE splicer.end(BridgeTest.Com._includes)

// user defined constructor
void BridgeTest::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._ctor)
}

// user defined destructor
void BridgeTest::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._dtor)
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
BridgeTest::Com_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com.setServices)
  svc=services;
  BridgeTest::GoPort goPort=BridgeTest::GoPort::_create();
  goPort.setService(svc);

  svc.addProvidesPort(goPort,"go","gov.cca.ports.GoPort",0);
  svc.registerUsesPort("btport","gov.cca.ports.BridgeTestPort",0);
  // DO-NOT-DELETE splicer.end(BridgeTest.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(BridgeTest.Com._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(BridgeTest.Com._misc)

