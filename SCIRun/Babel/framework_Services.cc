// 
// File:          framework_Services.cc
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.0
// SIDL Created:  20020813 11:07:10 CDT
// Generated:     20020813 11:07:12 CDT
// Description:   Client-side glue code for framework.Services
// 
// WARNING: Automatically generated; changes will be lost
// 
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
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
void
framework::Services::addReference(  )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
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
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
framework::Services::deleteReference(  )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
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
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
framework::Services::isSame( /*in*/ SIDL::BaseInterface iobj )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
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
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL::BaseInterface
framework::Services::queryInterface( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::queryInterface()\""
    ));
  }
  SIDL::BaseInterface _result;
  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  _result = SIDL::BaseInterface( (*(d_self->d_epv->f_queryInterface))(d_self,
    /* in */ _local_name ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  SIDL_String_free( _local_name );
  return _result;
}



/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
bool
framework::Services::isInstanceOf( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::isInstanceOf()\""
    ));
  }
  bool _result;
  // pack args to dispatch to ior
  SIDL_bool _local_result;
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_isInstanceOf))(d_self,
    /* in */ _local_name );
  // unpack results and cleanup
  _result = (_local_result == TRUE);
  SIDL_String_free( _local_name );
  return _result;
}


/**
 * user defined non-static method.
 */
void*
framework::Services::getData(  )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
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
 * Ask for a previously registered Port; will return a Port or generate an error. 
 */
govcca::Port
framework::Services::getPort( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::getPort()\""
    ));
  }
  govcca::Port _result;
  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  _result = govcca::Port( (*(d_self->d_epv->f_getPort))(d_self,
    /* in */ _local_name ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  SIDL_String_free( _local_name );
  return _result;
}



/**
 * Ask for a previously registered Port and return that Port if it is
 * available or return null otherwise. 
 */
govcca::Port
framework::Services::getPortNonblocking( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::getPortNonblocking()\""
    ));
  }
  govcca::Port _result;
  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  _result = govcca::Port( (*(d_self->d_epv->f_getPortNonblocking))(d_self,
    /* in */ _local_name ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  SIDL_String_free( _local_name );
  return _result;
}



/**
 * Modified according to Motion 31 
 */
void
framework::Services::registerUsesPort( /*in*/ std::string name,
  /*in*/ std::string type, /*in*/ govcca::TypeMap properties )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::registerUsesPort()\""
    ));
  }

  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  char * _local_type = SIDL_String_strdup( type.c_str() );
  // dispatch to ior
  (*(d_self->d_epv->f_registerUsesPort))(d_self, /* in */ _local_name,
    /* in */ _local_type, /* in */ properties._get_ior() );
  // unpack results and cleanup
  SIDL_String_free( _local_name );
  SIDL_String_free( _local_type );
}



/**
 * Notify the framework that a Port, previously registered by this component,
 * is no longer needed. 
 */
void
framework::Services::unregisterUsesPort( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::unregisterUsesPort()\""
    ));
  }

  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  (*(d_self->d_epv->f_unregisterUsesPort))(d_self, /* in */ _local_name );
  // unpack results and cleanup
  SIDL_String_free( _local_name );
}



/**
 * Exports a Port implemented by this component to the framework.  
 * This Port is now available for the framework to connect to other components. 
 * Modified according to Motion 31 
 */
void
framework::Services::addProvidesPort( /*in*/ govcca::Port inPort,
  /*in*/ std::string name, /*in*/ std::string type,
  /*in*/ govcca::TypeMap properties )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::addProvidesPort()\""
    ));
  }

  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  char * _local_type = SIDL_String_strdup( type.c_str() );
  // dispatch to ior
  (*(d_self->d_epv->f_addProvidesPort))(d_self, /* in */ inPort._get_ior(),
    /* in */ _local_name, /* in */ _local_type,
    /* in */ properties._get_ior() );
  // unpack results and cleanup
  SIDL_String_free( _local_name );
  SIDL_String_free( _local_type );
}



/**
 * Notifies the framework that a previously exported Port is no longer 
 * available for use.
 */
void
framework::Services::removeProvidesPort( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::removeProvidesPort()\""
    ));
  }

  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  (*(d_self->d_epv->f_removeProvidesPort))(d_self, /* in */ _local_name );
  // unpack results and cleanup
  SIDL_String_free( _local_name );
}



/**
 * Notifies the framework that this component is finished with this Port.   
 * releasePort() method calls exactly match getPort() mehtod calls.  After 
 * releasePort() is invoked all references to the released Port become invalid. 
 */
void
framework::Services::releasePort( /*in*/ std::string name )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::releasePort()\""
    ));
  }

  // pack args to dispatch to ior
  char * _local_name = SIDL_String_strdup( name.c_str() );
  // dispatch to ior
  (*(d_self->d_epv->f_releasePort))(d_self, /* in */ _local_name );
  // unpack results and cleanup
  SIDL_String_free( _local_name );
}


/**
 * user defined non-static method.
 */
govcca::TypeMap
framework::Services::createTypeMap(  )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::createTypeMap()\""
    ));
  }
  govcca::TypeMap _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = govcca::TypeMap( (*(d_self->d_epv->f_createTypeMap))(d_self ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}


/**
 * user defined non-static method.
 */
govcca::TypeMap
framework::Services::getPortProperties( /*in*/ std::string portName )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::getPortProperties()\""
    ));
  }
  govcca::TypeMap _result;
  // pack args to dispatch to ior
  char * _local_portName = SIDL_String_strdup( portName.c_str() );
  // dispatch to ior
  _result = govcca::TypeMap( (*(d_self->d_epv->f_getPortProperties))(d_self,
    /* in */ _local_portName ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  SIDL_String_free( _local_portName );
  return _result;
}



/**
 * Get a reference to the component to which this Services object belongs. 
 */
govcca::ComponentID
framework::Services::getComponentID(  )
throw ( SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw SIDL::NullIORException( std::string (
      "Null IOR Pointer in \"framework::Services::getComponentID()\""
    ));
  }
  govcca::ComponentID _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = govcca::ComponentID( (*(d_self->d_epv->f_getComponentID))(d_self ));
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
framework::Services
framework::Services::_create() {
  framework::Services self( (*_get_ext()->createObject)() );
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
framework::Services::Services ( const framework::Services& original ) {
  d_self = const_cast<ior_t*>(original.d_self);
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addReference();
  }
}

// assignment operator
framework::Services&
framework::Services::operator=( const framework::Services& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteReference();
    }
    d_self = const_cast<ior_t*>(rhs.d_self);
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addReference();
    }
  }
  return *this;
}

// conversion from ior to C++ class
framework::Services::Services ( framework::Services::ior_t* ior ) 
    : d_self( ior ), d_weak_reference(false) {
  if ( d_self != 0 ) {
    addReference();
  }
}

// Alternate constructor: does not call addReference()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
framework::Services::Services ( framework::Services::ior_t* ior, bool isWeak ) 
    : d_self( ior ), d_weak_reference(isWeak) { 
}

// conversion from a StubBase
framework::Services::Services ( const SIDL::StubBase& base )
{
  d_self = reinterpret_cast<ior_t*>(base._cast("framework.Services"));
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
    ptr = reinterpret_cast<void*>((*d_self->d_epv->f__cast)(d_self, type));
  }
  return ptr;
}

// Static data type
const framework::Services::ext_t * framework::Services::s_ext;

// private static method to get static data type
const framework::Services::ext_t *
framework::Services::_get_ext()
  throw (SIDL::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = framework_Services__externals();
#else
    const ext_t *(*dll_f)(void) =
      (const ext_t *(*)(void)) SIDL::Loader::lookupSymbol(
        "framework_Services__externals");
    s_ext = (dll_f ? (*dll_f)() : NULL);
    if (!s_ext) {
      throw SIDL::NullIORException( std::string (
        "cannot find implementation for framework.Services; please set SIDL_DLL_PATH"
      ));
    }
#endif
  }
  return s_ext;
}

