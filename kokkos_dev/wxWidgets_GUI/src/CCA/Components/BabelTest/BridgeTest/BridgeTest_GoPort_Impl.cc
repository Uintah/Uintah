// 
// File:          BridgeTest_GoPort_Impl.cc
// Symbol:        BridgeTest.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for BridgeTest.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "BridgeTest_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._includes)
#include "BridgeTest.hh"
#include <iostream>
// DO-NOT-DELETE splicer.end(BridgeTest.GoPort._includes)

// user-defined constructor.
void BridgeTest::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._ctor)
  // Insert-Code-Here {BridgeTest.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort._ctor)
}

// user-defined destructor.
void BridgeTest::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._dtor)
  // Insert-Code-Here {BridgeTest.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort._dtor)
}

// static class initializer.
void BridgeTest::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._load)
  // Insert-Code-Here {BridgeTest.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setServices[]
 */
void
BridgeTest::GoPort_impl::setServices (
  /* in */ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(BridgeTest.GoPort.setServices)
  this->svc = svc;
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort.setServices)
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
  BridgeTest::BridgeTestPort s = svc.getPort("btport");
  std::cerr << "Got the port" << std::endl;
  if(!s._is_nil()) {
    std::cerr << "Sent " << std::endl;//<< s.m2(13) <<"\n";
  } else {
    std::cerr << "getPort() returns null" << std::endl;
  }
  return 0;
  // DO-NOT-DELETE splicer.end(BridgeTest.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(BridgeTest.GoPort._misc)
// Insert-Code-Here {BridgeTest.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(BridgeTest.GoPort._misc)

