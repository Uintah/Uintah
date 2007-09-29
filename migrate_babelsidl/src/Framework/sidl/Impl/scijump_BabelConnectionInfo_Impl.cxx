// 
// File:          scijump_BabelConnectionInfo_Impl.cxx
// Symbol:        scijump.BabelConnectionInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.BabelConnectionInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_BabelConnectionInfo_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_core_ComponentInfo_hxx
#include "sci_cca_core_ComponentInfo.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._includes)

#include "scijump.hxx"

#include <Core/Thread/Guard.h>

using namespace SCIRun;

// DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::BabelConnectionInfo_impl::BabelConnectionInfo_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::BabelConnectionInfo::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._ctor2)
  // Insert-Code-Here {scijump.BabelConnectionInfo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._ctor2)
}

// user defined constructor
void scijump::BabelConnectionInfo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._ctor)
  valid = false;
  lock = new Mutex("BabelConnectionInfo lock");
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._ctor)
}

// user defined destructor
void scijump::BabelConnectionInfo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._dtor)
  delete lock;
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._dtor)
}

// static class initializer
void scijump::BabelConnectionInfo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._load)
  // Insert-Code-Here {scijump.BabelConnectionInfo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::BabelConnectionInfo_impl::initialize_impl (
  /* in */::sci::cca::core::ComponentInfo& user,
  /* in */::sci::cca::core::ComponentInfo& provider,
  /* in */const ::std::string& userPortName,
  /* in */const ::std::string& providerPortName,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.initialize)

  Guard g(lock);
  this->user = user;
  this->provider = provider;
  this->userPortName = userPortName;
  this->providerPortName = providerPortName;
  if (properties._is_nil()) {
    this->properties = scijump::TypeMap::_create();
  } else {
    this->properties = properties;
  }
  valid = true;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.initialize)
}

/**
 * Method:  getProperties[]
 */
::gov::cca::TypeMap
scijump::BabelConnectionInfo_impl::getProperties_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.getProperties)
  return properties;
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.getProperties)
}

/**
 * Method:  setProperties[]
 */
void
scijump::BabelConnectionInfo_impl::setProperties_impl (
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.setProperties)

  Guard g(lock);
  // allow null properties?
  if (properties._not_nil()) {
    this->properties = properties;
  }

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.setProperties)
}

/**
 * Method:  invalidate[]
 */
void
scijump::BabelConnectionInfo_impl::invalidate_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.invalidate)

  Guard g(lock);
  user = 0;
  provider = 0;
  userPortName.clear();
  providerPortName.clear();
  properties = 0;
  valid = false;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.invalidate)
}

/**
 *  
 * Get the providing component (callee) ID.
 * @return ComponentID of the component that has 
 * provided the Port for this connection. 
 * @throws CCAException if the underlying connection 
 * is no longer valid.
 */
::gov::cca::ComponentID
scijump::BabelConnectionInfo_impl::getProvider_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.getProvider)

  if (provider._is_nil() || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getProvider");
    throw ex;
  } 
  return provider;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.getProvider)
}

/**
 *  
 * Get the using component (caller) ID.
 * @return ComponentID of the component that is using the provided Port.
 * @throws CCAException if the underlying connection is no longer valid.
 */
::gov::cca::ComponentID
scijump::BabelConnectionInfo_impl::getUser_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.getUser)

  if (user._is_nil() || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getUser");
    throw ex;
  } 
  return user;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.getUser)
}

/**
 *  
 * Get the port name in the providing component of this connection.
 * @return the instance name of the provided Port.
 * @throws CCAException if the underlying connection is no longer valid.
 */
::std::string
scijump::BabelConnectionInfo_impl::getProviderPortName_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.getProviderPortName)

  if (providerPortName.size() < 1 || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getProviderPortName");
    throw ex;
  } 
  return providerPortName;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.getProviderPortName)
}

/**
 *  
 * Get the port name in the using component of this connection.
 * Return the instance name of the Port registered for use in 
 * this connection.
 * @throws CCAException if the underlying connection is no longer valid.
 */
::std::string
scijump::BabelConnectionInfo_impl::getUserPortName_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo.getUserPortName)

  if (userPortName.size() < 1 || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getUserPortName");
    throw ex;
  } 
  return userPortName;

  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo.getUserPortName)
}


// DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._misc)
// Insert-Code-Here {scijump.BabelConnectionInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._misc)

