// 
// File:          framework_Services.cc
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030306 10:46:17 MST
// Generated:     20030306 10:46:22 MST
// Description:   Client-side glue code for framework.Services
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/cca-debug/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_Services_hh
#include "framework_Services.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_SIDL_BaseClass_hh
#include "SIDL_BaseClass.hh"
#endif
#ifndef included_SIDL_BaseException_hh
#include "SIDL_BaseException.hh"
#endif
#include "SIDL_String.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "SIDL_Loader.hh"
#endif


//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * &amp;lt;p&amp;gt;
 * Add one to the intrinsic reference count in the underlying object.
 * Object in &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * &amp;lt;/p&amp;gt;
 * &amp;lt;p&amp;gt;
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * &amp;lt;/p&amp;gt;
 */
void
framework::Services::addReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::addReference()\""
    ));
  }

  if ( !d_weak_reference ) {
    // pack args to dispatch to ior

    // dispatch to ior
    (*(d_self->d_epv->f_addReference))(d_self );
    // unpack results and cleanup

  }
}



/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
framework::Services::deleteReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::deleteReference()\""
    ));
  }

  if ( !d_weak_reference ) {
    // pack args to dispatch to ior

    // dispatch to ior
    (*(d_self->d_epv->f_deleteReference))(d_self );
    // unpack results and cleanup

    d_self = 0;
  }
}



/**
 * Return true if and only if &amp;lt;code&amp;gt;obj&amp;lt;/code&amp;gt; refers to the same
 * object as this object.
 */
bool
framework::Services::isSame( /*in*/ ::SIDL::BaseInterface iobj )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::isSame()\""
    ));
  }
  bool _result;
  // pack args to dispatch to ior
  SIDL_bool _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_isSame))(d_self,
    /* in */ iobj._get_ior() );
  // unpack results and cleanup
  _result = (_local_result == TRUE);
  return _result;
}



/**
 * Check whether the object can support the specified interface or
 * class.  If the &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; type name in &amp;lt;code&amp;gt;name&amp;lt;/code&amp;gt;
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling &amp;lt;code&amp;gt;deleteReference&amp;lt;/code&amp;gt; on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
::SIDL::BaseInterface
framework::Services::queryInterface( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::queryInterface()\""
    ));
  }
  ::SIDL::BaseInterface _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = ::SIDL::BaseInterface( (*(d_self->d_epv->f_queryInterface))(d_self,
    /* in */ name.c_str() ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; type name.  This
 * routine will return &amp;lt;code&amp;gt;true&amp;lt;/code&amp;gt; if and only if a cast to
 * the string type name would succeed.
 */
bool
framework::Services::isInstanceOf( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::isInstanceOf()\""
    ));
  }
  bool _result;
  // pack args to dispatch to ior
  SIDL_bool _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_isInstanceOf))(d_self,
    /* in */ name.c_str() );
  // unpack results and cleanup
  _result = (_local_result == TRUE);
  return _result;
}


/**
 * user defined non-static method.
 */
void*
framework::Services::getData(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::getData()\""
    ));
  }
  void* _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = (*(d_self->d_epv->f_getData))(d_self );
  // unpack results and cleanup

  return _result;
}



/**
 * Fetch a previously registered Port (defined by either 
 * addProvidePort or (more typically) registerUsesPort).  
 * @return Will return the Port (possibly waiting forever while
 * attempting to acquire it) or throw an exception. Does not return
 * NULL, even in the case where no connection has been made. 
 * If a Port is returned,
 * there is then a contract that the port will remain valid for use
 * by the caller until the port is released via releasePort(), or a 
 * Disconnect Event is successfully dispatched to the caller,
 * or a runtime exception (such as network failure) occurs during 
 * invocation of some function in the Port. 
 * &lt;p&gt;
 * Subtle interpretation: If the Component is not listening for
 * Disconnect events, then the framework has no clean way to
 * break the connection until after the component calls releasePort.
 * &lt;/p&gt;
 * &lt;p&gt;The framework may go through some machinations to obtain
 *    the port, possibly involving an interactive user or network 
 *    queries, before giving up and throwing an exception.
 * &lt;/p&gt;
 * 
 * @param portName The previously registered or provide port which
 * 	   the component now wants to use.
 * @exception CCAException with the following types: NotConnected, PortNotDefined, 
 *                NetworkError, OutOfMemory.
 */
::gov::cca::Port
framework::Services::getPort( /*in*/ const ::std::string& portName )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::getPort()\""
    ));
  }
  ::gov::cca::Port _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = ::gov::cca::Port( (*(d_self->d_epv->f_getPort))(d_self,
    /* in */ portName.c_str(), &_exception ));
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Get a previously registered Port (defined by
 * either addProvide or registerUses) and return that
 * Port if it is available immediately (already connected
 * without further connection machinations).
 * There is an contract that the
 * port will remain valid per the description of getPort.
 * @return The named port, if it exists and is connected or self-provided,
 * 	      or NULL if it is registered and is not yet connected. Does not
 * 	      return if the Port is neither registered nor provided, but rather
 * 	      throws an exception.
 * @param portName registered or provided port that
 * 	     the component now wants to use.
 * @exception CCAException with the following types: PortNotDefined, OutOfMemory.
 */
::gov::cca::Port
framework::Services::getPortNonblocking( /*in*/ const ::std::string& portName )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::getPortNonblocking()\""
    ));
  }
  ::gov::cca::Port _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = ::gov::cca::Port( (*(d_self->d_epv->f_getPortNonblocking))(d_self,
    /* in */ portName.c_str(), &_exception ));
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Notifies the framework that this component is finished 
 * using the previously fetched Port that is named.     
 * The releasePort() method calls should be paired with 
 * getPort() method calls; however, an extra call to releasePort()
 * for the same name may (is not required to) generate an exception.
 * Calls to release ports which are not defined or have never be fetched
 * with one of the getPort functions generate exceptions.
 * @param portName The name of a port.
 * @exception CCAException with the following types: PortNotDefined, PortNotInUse.
 */
void
framework::Services::releasePort( /*in*/ const ::std::string& portName )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::releasePort()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  (*(d_self->d_epv->f_releasePort))(d_self, /* in */ portName.c_str(),
    &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }

}



/**
 * Creates a TypeMap, potentially to be used in subsequent
 * calls to describe a Port.  Initially, this map is empty.
 */
::gov::cca::TypeMap
framework::Services::createTypeMap(  )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::createTypeMap()\""
    ));
  }
  ::gov::cca::TypeMap _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = ::gov::cca::TypeMap( (*(d_self->d_epv->f_createTypeMap))(d_self,
    &_exception ));
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Register a request for a Port that will be retrieved subsequently 
 * with a call to getPort().
 * @param portName A string uniquely describing this port.  This string
 * must be unique for this component, over both uses and provides ports.
 * @param type A string desribing the type of this port.
 * @param properties A TypeMap describing optional properties
 * associated with this port. This can be a null pointer, which
 * indicates an empty list of properties.  Properties may be
 * obtained from createTypeMap or any other source.  The properties
 * be copied into the framework, and subsequent changes to the
 * properties object will have no effect on the properties
 * associated with this port.
 * In these properties, all frameworks recognize at least the
 * following keys and values in implementing registerUsesPort:
 * &lt;pre&gt;
 * key:              standard values (in string form)     default
 * &quot;MAX_CONNECTIONS&quot; any nonnegative integer, &quot;unlimited&quot;.   1
 * &quot;MIN_CONNECTIONS&quot; any integer &gt; 0.                        0
 * &quot;ABLE_TO_PROXY&quot;   &quot;true&quot;, &quot;false&quot;                      &quot;false&quot;
 * &lt;/pre&gt;
 * The component is not expected to work if the framework
 * has not satisfied the connection requirements.
 * The framework is allowed to return an error if it
 * is incapable of meeting the connection requirements,
 * e.g. it does not implement multiple uses ports.
 * The caller of registerUsesPort is not obligated to define
 * these properties. If left undefined, the default listed above is
 *       assumed.
 * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
 */
void
framework::Services::registerUsesPort( /*in*/ const ::std::string& portName,
  /*in*/ const ::std::string& type, /*in*/ ::gov::cca::TypeMap properties )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::registerUsesPort()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  (*(d_self->d_epv->f_registerUsesPort))(d_self, /* in */ portName.c_str(),
    /* in */ type.c_str(), /* in */ properties._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }

}



/**
 * Notify the framework that a Port, previously registered by this
 * component but currently not in use, is no longer desired. 
 * Unregistering a port that is currently 
 * in use (i.e. an unreleased getPort() being outstanding) 
 * is an error.
 * @param name The name of a registered Port.
 * @exception CCAException with the following types: UsesPortNotReleased, PortNotDefined.
 */
void
framework::Services::unregisterUsesPort( /*in*/ const ::std::string& portName )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::unregisterUsesPort()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  (*(d_self->d_epv->f_unregisterUsesPort))(d_self, /* in */ portName.c_str(),
    &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }

}



/**
 * Exposes a Port from this component to the framework.  
 * This Port is now available for the framework to connect 
 * to other components. 
 * @param inPort An abstract interface (tagged with CCA-ness
 * 	by inheriting from gov.cca.Port) the framework will
 * 	make available to other components.
 * 
 * @param portName string uniquely describing this port.  This string
 * must be unique for this component, over both uses and provides ports.
 * 
 * @param type string describing the type (class) of this port.
 * 
 * @param properties A TypeMap describing optional properties
 * associated with this port. This can be a null pointer, which
 * indicates an empty list of properties.  Properties may be
 * obtained from createTypeMap or any other source.  The properties
 * be copied into the framework, and subsequent changes to the
 * properties object will have no effect on the properties
 * associated with this port.
 * In these properties, all frameworks recognize at least the
 * following keys and values in implementing registerUsesPort:
 * &lt;pre&gt;
 * key:              standard values (in string form)     default
 * &quot;MAX_CONNECTIONS&quot; any nonnegative integer, &quot;unlimited&quot;.   1
 * &quot;MIN_CONNECTIONS&quot; any integer &gt; 0.                        0
 * &quot;ABLE_TO_PROXY&quot;   &quot;true&quot;, &quot;false&quot;                      &quot;false&quot;
 * &lt;/pre&gt;
 * The component is not expected to work if the framework
 * has not satisfied the connection requirements.
 * The framework is allowed to return an error if it
 * is incapable of meeting the connection requirements,
 * e.g. it does not implement multiple uses ports.
 * The caller of addProvidesPort is not obligated to define
 * these properties. If left undefined, the default listed above is
 * assumed.
 * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
 */
void
framework::Services::addProvidesPort( /*in*/ ::gov::cca::Port inPort,
  /*in*/ const ::std::string& portName, /*in*/ const ::std::string& type,
  /*in*/ ::gov::cca::TypeMap properties )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::addProvidesPort()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  (*(d_self->d_epv->f_addProvidesPort))(d_self, /* in */ inPort._get_ior(),
    /* in */ portName.c_str(), /* in */ type.c_str(),
    /* in */ properties._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }

}



/**
 * Returns the complete list of the properties for a Port.  This
 * includes the properties defined when the port was registered
 * (these properties can be modified by the framework), two special
 * properties &quot;cca.portName&quot; and &quot;cca.portType&quot;, and any other
 * properties that the framework wishes to disclose to the component.
 * The framework may also choose to provide only the subset of input
 * properties (i.e. from addProvidesPort/registerUsesPort) that it
 * will honor.      
 */
::gov::cca::TypeMap
framework::Services::getPortProperties( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::getPortProperties()\""
    ));
  }
  ::gov::cca::TypeMap _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = ::gov::cca::TypeMap( (*(d_self->d_epv->f_getPortProperties))(d_self,
    /* in */ name.c_str() ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Notifies the framework that a previously exposed Port is no longer 
 * available for use. The Port being removed must exist
 * until this call returns, or a CCAException may occur.
 * @param name The name of a provided Port.
 * @exception PortNotDefined. In general, the framework will not dictate 
 * when the component chooses to stop offering services.
 */
void
framework::Services::removeProvidesPort( /*in*/ const ::std::string& portName )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::CCAException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::removeProvidesPort()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  (*(d_self->d_epv->f_removeProvidesPort))(d_self, /* in */ portName.c_str(),
    &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.CCAException")) != 0 ) {
      struct gov_cca_CCAException__object * _realtype = reinterpret_cast< 
        struct gov_cca_CCAException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::CCAException( _realtype, false );
    }
  }

}



/**
 * Get a reference to the component to which this 
 * Services object belongs. 
 */
::gov::cca::ComponentID
framework::Services::getComponentID(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::Services::getComponentID()\""
    ));
  }
  ::gov::cca::ComponentID _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = ::gov::cca::ComponentID( 
    (*(d_self->d_epv->f_getComponentID))(d_self ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::framework::Services
framework::Services::_create() {
  ::framework::Services self( (*_get_ext()->createObject)() );
  // NOTE: reference count == 2. 
  //   (1 from createObject, 1 from IOR->C++)
  // Decrement this count back down to one.
  (*(self.d_self->d_epv->f_deleteReference))(self.d_self);
  return self;
}

// default destructor
framework::Services::~Services () {
  if ( d_self != 0 ) {
    deleteReference();
  }
}

// copy constructor
framework::Services::Services ( const ::framework::Services& original ) {
  d_self = const_cast< ior_t*>(original.d_self);
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addReference();
  }
}

// assignment operator
::framework::Services&
framework::Services::operator=( const ::framework::Services& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteReference();
    }
    d_self = const_cast< ior_t*>(rhs.d_self);
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addReference();
    }
  }
  return *this;
}

// conversion from ior to C++ class
framework::Services::Services ( ::framework::Services::ior_t* ior ) 
    : d_self( ior ), d_weak_reference(false) {
  if ( d_self != 0 ) {
    addReference();
  }
}

// Alternate constructor: does not call addReference()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
framework::Services::Services ( ::framework::Services::ior_t* ior,
  bool isWeak ) 
    : d_self( ior ), d_weak_reference(isWeak) { 
}

// conversion from a StubBase
framework::Services::Services ( const ::SIDL::StubBase& base )
{
  d_self = reinterpret_cast< ior_t*>(base._cast("framework.Services"));
  d_weak_reference = false;
  if (d_self != 0) {
    addReference();
  }
}

// protected method that implements casting
void* framework::Services::_cast(const char* type) const
{
  void* ptr = 0;
  if ( d_self != 0 ) {
    ptr = reinterpret_cast< void*>((*d_self->d_epv->f__cast)(d_self, type));
  }
  return ptr;
}

// Static data type
const ::framework::Services::ext_t * framework::Services::s_ext;

// private static method to get static data type
const ::framework::Services::ext_t *
framework::Services::_get_ext()
  throw (::SIDL::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = framework_Services__externals();
#else
    const ext_t *(*dll_f)(void) =
      (const ext_t *(*)(void)) ::SIDL::Loader::lookupSymbol(
        "framework_Services__externals");
    s_ext = (dll_f ? (*dll_f)() : NULL);
    if (!s_ext) {
      throw ::SIDL::NullIORException( ::std::string (
        "cannot find implementation for framework.Services; please set SIDL_DLL_PATH"
      ));
    }
#endif
  }
  return s_ext;
}

