// 
// File:          BridgeTest_BridgeTestPort_Impl.cxx
// Symbol:        BridgeTest.BridgeTestPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for BridgeTest.BridgeTestPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "BridgeTest_BridgeTestPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._includes)
// Insert-Code-Here {BridgeTest.BridgeTestPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
BridgeTest::BridgeTestPort_impl::BridgeTestPort_impl() : StubBase(
  reinterpret_cast< void*>(::BridgeTest::BridgeTestPort::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._ctor2)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._ctor2)
}

// user defined constructor
void BridgeTest::BridgeTestPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._ctor)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._ctor)
}

// user defined destructor
void BridgeTest::BridgeTestPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._dtor)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._dtor)
}

// static class initializer
void BridgeTest::BridgeTestPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._load)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  m2[]
 */
void
BridgeTest::BridgeTestPort_impl::m2_impl (
  /* in array<int> */::sidl::array<int32_t>& a ) 
{
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort.m2)
  // Insert-Code-Here {BridgeTest.BridgeTestPort.m2} (m2 method)
  // 
  // This method has not been implemented
  // 
    ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
    ex.setNote("This method has not been implemented");
    ex.add(__FILE__, __LINE__, "m2");
    throw ex;
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort.m2)
}


// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._misc)
// Insert-Code-Here {BridgeTest.BridgeTestPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._misc)

