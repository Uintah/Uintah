// 
// File:          whoc_IDPort.cc
// Symbol:        whoc.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030305 18:50:02 MST
// Generated:     20030305 18:50:07 MST
// Description:   Client-side glue code for whoc.IDPort
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
// 

#ifndef included_whoc_IDPort_hh
#include "whoc_IDPort.hh"
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
whoc::IDPort::addReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::addReference()\""
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
whoc::IDPort::deleteReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::deleteReference()\""
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
whoc::IDPort::isSame( /*in*/ ::SIDL::BaseInterface iobj )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::isSame()\""
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
whoc::IDPort::queryInterface( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::queryInterface()\""
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
whoc::IDPort::isInstanceOf( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::isInstanceOf()\""
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
 * Test prot. Return a string as an ID for Hello component
 */
::std::string
whoc::IDPort::getID(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"whoc::IDPort::getID()\""
    ));
  }
  ::std::string _result;
  // pack args to dispatch to ior
  char * _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getID))(d_self );
  // unpack results and cleanup
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
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
::whoc::IDPort
whoc::IDPort::_create() {
  ::whoc::IDPort self( (*_get_ext()->createObject)() );
  // NOTE: reference count == 2. 
  //   (1 from createObject, 1 from IOR->C++)
  // Decrement this count back down to one.
  (*(self.d_self->d_epv->f_deleteReference))(self.d_self);
  return self;
}

// default destructor
whoc::IDPort::~IDPort () {
  if ( d_self != 0 ) {
    deleteReference();
  }
}

// copy constructor
whoc::IDPort::IDPort ( const ::whoc::IDPort& original ) {
  d_self = const_cast< ior_t*>(original.d_self);
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addReference();
  }
}

// assignment operator
::whoc::IDPort&
whoc::IDPort::operator=( const ::whoc::IDPort& rhs ) {
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
whoc::IDPort::IDPort ( ::whoc::IDPort::ior_t* ior ) 
    : d_self( ior ), d_weak_reference(false) {
  if ( d_self != 0 ) {
    addReference();
  }
}

// Alternate constructor: does not call addReference()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
whoc::IDPort::IDPort ( ::whoc::IDPort::ior_t* ior, bool isWeak ) 
    : d_self( ior ), d_weak_reference(isWeak) { 
}

// conversion from a StubBase
whoc::IDPort::IDPort ( const ::SIDL::StubBase& base )
{
  d_self = reinterpret_cast< ior_t*>(base._cast("whoc.IDPort"));
  d_weak_reference = false;
  if (d_self != 0) {
    addReference();
  }
}

// protected method that implements casting
void* whoc::IDPort::_cast(const char* type) const
{
  void* ptr = 0;
  if ( d_self != 0 ) {
    ptr = reinterpret_cast< void*>((*d_self->d_epv->f__cast)(d_self, type));
  }
  return ptr;
}

// Static data type
const ::whoc::IDPort::ext_t * whoc::IDPort::s_ext;

// private static method to get static data type
const ::whoc::IDPort::ext_t *
whoc::IDPort::_get_ext()
  throw (::SIDL::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = whoc_IDPort__externals();
#else
    const ext_t *(*dll_f)(void) =
      (const ext_t *(*)(void)) ::SIDL::Loader::lookupSymbol(
        "whoc_IDPort__externals");
    s_ext = (dll_f ? (*dll_f)() : NULL);
    if (!s_ext) {
      throw ::SIDL::NullIORException( ::std::string (
        "cannot find implementation for whoc.IDPort; please set SIDL_DLL_PATH"
      ));
    }
#endif
  }
  return s_ext;
}

