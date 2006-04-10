// 
// File:          BridgeTest_Com_Impl.cc
// Symbol:        BridgeTest.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for BridgeTest.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "BridgeTest_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(BridgeTest.Com._includes)
#include "BridgeTest.hh"
// DO-NOT-DELETE splicer.end(BridgeTest.Com._includes)

// user-defined constructor.
void BridgeTest::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._ctor)
  // Insert-Code-Here {BridgeTest.Com._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._ctor)
}

// user-defined destructor.
void BridgeTest::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._dtor)
  // Insert-Code-Here {BridgeTest.Com._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._dtor)
}

// static class initializer.
void BridgeTest::Com_impl::_load() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._load)
  // Insert-Code-Here {BridgeTest.Com._load} (class initialization)
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Starts up a component presence in the calling framework.
 * @param Svc the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */
void
BridgeTest::Com_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com.setServices)
  svc = services;
  BridgeTest::GoPort goPort = BridgeTest::GoPort::_create();
  goPort.setServices(svc);

  svc.addProvidesPort(goPort, "go", "gov.cca.ports.GoPort", 0);
  svc.registerUsesPort("btport", "gov.cca.ports.BridgeTestPort", 0);
  // DO-NOT-DELETE splicer.end(BridgeTest.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(BridgeTest.Com._misc)
// Insert-Code-Here {BridgeTest.Com._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(BridgeTest.Com._misc)

