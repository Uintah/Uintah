// 
// File:          framework_ComponentID_Impl.cxx
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "framework_ComponentID_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(framework.ComponentID._includes)
// Insert-Code-Here {framework.ComponentID._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(framework.ComponentID._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
framework::ComponentID_impl::ComponentID_impl() : StubBase(reinterpret_cast< 
  void*>(::framework::ComponentID::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._ctor2)
  // insert code here (ctor2)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._ctor2)
}

// user defined constructor
void framework::ComponentID_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._ctor)
  // Insert-Code-Here {framework.ComponentID._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._ctor)
}

// user defined destructor
void framework::ComponentID_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._dtor)
  // Insert-Code-Here {framework.ComponentID._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._dtor)
}

// static class initializer
void framework::ComponentID_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._load)
  // Insert-Code-Here {framework.ComponentID._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Returns the instance name provided in
 * <code>BuilderService.createInstance()</code>
 * or in
 * <code>AbstractFramework.getServices()</code>.
 * @throws CCAException if <code>ComponentID</code> is invalid
 */
::std::string
framework::ComponentID_impl::getInstanceName_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(framework.ComponentID.getInstanceName)
  // Insert-Code-Here {framework.ComponentID.getInstanceName} (getInstanceName method)
  // DO-NOT-DELETE splicer.end(framework.ComponentID.getInstanceName)
}

/**
 * Returns a framework specific serialization of the ComponentID.
 * @throws CCAException if <code>ComponentID</code> is
 * invalid.
 */
::std::string
framework::ComponentID_impl::getSerialization_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(framework.ComponentID.getSerialization)
  // Insert-Code-Here {framework.ComponentID.getSerialization} (getSerialization method)
  // DO-NOT-DELETE splicer.end(framework.ComponentID.getSerialization)
}


// DO-NOT-DELETE splicer.begin(framework.ComponentID._misc)
// Insert-Code-Here {framework.ComponentID._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(framework.ComponentID._misc)

