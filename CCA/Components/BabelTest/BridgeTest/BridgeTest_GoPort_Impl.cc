// 
// File:          BridgeTest_GoPort_Impl.cc
// Symbol:        BridgeTest.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040412 13:28:06 MST
// Generated:     20040412 13:28:11 MST
// Description:   Server-side implementation for BridgeTest.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 6
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/BridgeTest/BridgeTest.sidl
// 
#include "BridgeTest_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._includes)
#include "BridgeTest.hh"
#include <iostream>
// DO-NOT-DELETE splicer.end(BridgeTest.GoPort._includes)

// user defined constructor
void BridgeTest::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort._ctor)
}

// user defined destructor
void BridgeTest::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setService[]
 */
void
BridgeTest::GoPort_impl::setService (
  /*in*/ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort.setService)
  this->svc=svc;
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort.setService)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
BridgeTest::GoPort_impl::go () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort.go)
  BridgeTest::BridgeTestPort s=svc.getPort("btport");
  std::cerr << "Got the port\n";
  if(!s._is_nil()) std::cerr<<"Sent ";//<< s.m2(13) <<"\n";
  else std::cerr<<"getPort() returns null\n";
  return 0;
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(BridgeTest.GoPort._misc)

