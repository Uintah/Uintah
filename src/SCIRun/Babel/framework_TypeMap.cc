// 
// File:          framework_TypeMap.cc
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030306 10:46:20 MST
// Generated:     20030306 10:46:22 MST
// Description:   Client-side glue code for framework.TypeMap
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.7.4
// source-line   = 17
// source-url    = file:/home/sci/kzhang/SCIRun/cca-debug/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_TypeMap_hh
#include "framework_TypeMap.hh"
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
framework::TypeMap::addReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::addReference()\""
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
framework::TypeMap::deleteReference(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::deleteReference()\""
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
framework::TypeMap::isSame( /*in*/ ::SIDL::BaseInterface iobj )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::isSame()\""
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
framework::TypeMap::queryInterface( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::queryInterface()\""
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
framework::TypeMap::isInstanceOf( /*in*/ const ::std::string& name )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::isInstanceOf()\""
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
 * Create an exact copy of this Map 
 */
::gov::cca::TypeMap
framework::TypeMap::cloneTypeMap(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::cloneTypeMap()\""
    ));
  }
  ::gov::cca::TypeMap _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = ::gov::cca::TypeMap( (*(d_self->d_epv->f_cloneTypeMap))(d_self ));
  // unpack results and cleanup
  if (_result._not_nil()) {
    // IOR return and constructor both increment, only need one
    
    (*(_result._get_ior()->d_epv->f_deleteReference))(_result._get_ior(
    )->d_object);
  }
  return _result;
}



/**
 * Create a new Map with no key/value associations. 
 */
::gov::cca::TypeMap
framework::TypeMap::cloneEmpty(  )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::cloneEmpty()\""
    ));
  }
  ::gov::cca::TypeMap _result;
  // pack args to dispatch to ior

  // dispatch to ior
  _result = ::gov::cca::TypeMap( (*(d_self->d_epv->f_cloneEmpty))(d_self ));
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
int32_t
framework::TypeMap::getInt( /*in*/ const ::std::string& key,
  /*in*/ int32_t dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getInt()\""
    ));
  }
  int32_t _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = (*(d_self->d_epv->f_getInt))(d_self, /* in */ key.c_str(),
    /* in */ dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }

  return _result;
}


/**
 * user defined non-static method.
 */
int64_t
framework::TypeMap::getLong( /*in*/ const ::std::string& key,
  /*in*/ int64_t dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getLong()\""
    ));
  }
  int64_t _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = (*(d_self->d_epv->f_getLong))(d_self, /* in */ key.c_str(),
    /* in */ dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }

  return _result;
}


/**
 * user defined non-static method.
 */
float
framework::TypeMap::getFloat( /*in*/ const ::std::string& key,
  /*in*/ float dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getFloat()\""
    ));
  }
  float _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = (*(d_self->d_epv->f_getFloat))(d_self, /* in */ key.c_str(),
    /* in */ dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }

  return _result;
}


/**
 * user defined non-static method.
 */
double
framework::TypeMap::getDouble( /*in*/ const ::std::string& key,
  /*in*/ double dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getDouble()\""
    ));
  }
  double _result;
  // pack args to dispatch to ior
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _result = (*(d_self->d_epv->f_getDouble))(d_self, /* in */ key.c_str(),
    /* in */ dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }

  return _result;
}


/**
 * user defined non-static method.
 */
::std::complex<float>
framework::TypeMap::getFcomplex( /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<float>& dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getFcomplex()\""
    ));
  }
  ::std::complex<float> _result;
  // pack args to dispatch to ior
  struct SIDL_fcomplex _local_result;
  struct SIDL_fcomplex _local_dflt = {dflt.real(), dflt.imag() } ; 
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getFcomplex))(d_self,
    /* in */ key.c_str(), /* in */ _local_dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result = reinterpret_cast< ::std::complex<float>&>(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::std::complex<double>
framework::TypeMap::getDcomplex( /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<double>& dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getDcomplex()\""
    ));
  }
  ::std::complex<double> _result;
  // pack args to dispatch to ior
  struct SIDL_dcomplex _local_result;
  struct SIDL_dcomplex _local_dflt = {dflt.real(), dflt.imag() } ; 
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getDcomplex))(d_self,
    /* in */ key.c_str(), /* in */ _local_dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result = reinterpret_cast< ::std::complex<double>&>(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::std::string
framework::TypeMap::getString( /*in*/ const ::std::string& key,
  /*in*/ const ::std::string& dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getString()\""
    ));
  }
  ::std::string _result;
  // pack args to dispatch to ior
  char * _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getString))(d_self, /* in */ key.c_str(),
    /* in */ dflt.c_str(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  return _result;
}


/**
 * user defined non-static method.
 */
bool
framework::TypeMap::getBool( /*in*/ const ::std::string& key, /*in*/ bool dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getBool()\""
    ));
  }
  bool _result;
  // pack args to dispatch to ior
  SIDL_bool _local_result;
  SIDL_bool _local_dflt = ( dflt!=  FALSE);
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getBool))(d_self, /* in */ key.c_str(),
    /* in */ _local_dflt, &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result = (_local_result == TRUE);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array<int>
framework::TypeMap::getIntArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<int> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getIntArray()\""
    ));
  }
  ::SIDL::array<int> _result;
  // pack args to dispatch to ior
  SIDL_int__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getIntArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array<long>
framework::TypeMap::getLongArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<long> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getLongArray()\""
    ));
  }
  ::SIDL::array<long> _result;
  // pack args to dispatch to ior
  SIDL_long__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getLongArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array<float>
framework::TypeMap::getFloatArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<float> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getFloatArray()\""
    ));
  }
  ::SIDL::array<float> _result;
  // pack args to dispatch to ior
  SIDL_float__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getFloatArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array<double>
framework::TypeMap::getDoubleArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<double> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getDoubleArray()\""
    ));
  }
  ::SIDL::array<double> _result;
  // pack args to dispatch to ior
  SIDL_double__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getDoubleArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array< ::SIDL::fcomplex>
framework::TypeMap::getFcomplexArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::fcomplex> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getFcomplexArray()\""
    ));
  }
  ::SIDL::array< ::SIDL::fcomplex> _result;
  // pack args to dispatch to ior
  SIDL_fcomplex__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getFcomplexArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array< ::SIDL::dcomplex>
framework::TypeMap::getDcomplexArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::dcomplex> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getDcomplexArray()\""
    ));
  }
  ::SIDL::array< ::SIDL::dcomplex> _result;
  // pack args to dispatch to ior
  SIDL_dcomplex__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getDcomplexArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array< ::std::string>
framework::TypeMap::getStringArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::std::string> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getStringArray()\""
    ));
  }
  ::SIDL::array< ::std::string> _result;
  // pack args to dispatch to ior
  SIDL_string__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getStringArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}


/**
 * user defined non-static method.
 */
::SIDL::array<bool>
framework::TypeMap::getBoolArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<bool> dflt )
throw ( 
  ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
)
{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getBoolArray()\""
    ));
  }
  ::SIDL::array<bool> _result;
  // pack args to dispatch to ior
  SIDL_bool__array* _local_result;
  SIDL_BaseException__object * _exception = 0;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getBoolArray))(d_self,
    /* in */ key.c_str(), /* in */ dflt._get_ior(), &_exception );
  // unpack results and cleanup
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception,
      "gov.cca.TypeMismatchException")) != 0 ) {
      struct gov_cca_TypeMismatchException__object * _realtype = 
        reinterpret_cast< struct gov_cca_TypeMismatchException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::gov::cca::TypeMismatchException( _realtype, false );
    }
  }
  _result._set_ior(_local_result);
  return _result;
}



/**
 * Assign a key and value. Any value previously assigned
 * to the same key will be overwritten.  
 */
void
framework::TypeMap::putInt( /*in*/ const ::std::string& key,
  /*in*/ int32_t value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putInt()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putInt))(d_self, /* in */ key.c_str(), /* in */ value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putLong( /*in*/ const ::std::string& key,
  /*in*/ int64_t value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putLong()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putLong))(d_self, /* in */ key.c_str(), /* in */ value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putFloat( /*in*/ const ::std::string& key,
  /*in*/ float value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putFloat()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putFloat))(d_self, /* in */ key.c_str(), /* in */ value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putDouble( /*in*/ const ::std::string& key,
  /*in*/ double value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putDouble()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putDouble))(d_self, /* in */ key.c_str(),
    /* in */ value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putFcomplex( /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<float>& value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putFcomplex()\""
    ));
  }

  // pack args to dispatch to ior
  struct SIDL_fcomplex _local_value = {value.real(), value.imag() } ;
  // dispatch to ior
  (*(d_self->d_epv->f_putFcomplex))(d_self, /* in */ key.c_str(),
    /* in */ _local_value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putDcomplex( /*in*/ const ::std::string& key,
  /*in*/ const ::std::complex<double>& value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putDcomplex()\""
    ));
  }

  // pack args to dispatch to ior
  struct SIDL_dcomplex _local_value = {value.real(), value.imag() } ;
  // dispatch to ior
  (*(d_self->d_epv->f_putDcomplex))(d_self, /* in */ key.c_str(),
    /* in */ _local_value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putString( /*in*/ const ::std::string& key,
  /*in*/ const ::std::string& value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putString()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putString))(d_self, /* in */ key.c_str(),
    /* in */ value.c_str() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putBool( /*in*/ const ::std::string& key,
  /*in*/ bool value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putBool()\""
    ));
  }

  // pack args to dispatch to ior
  SIDL_bool _local_value = ( value!=  FALSE);
  // dispatch to ior
  (*(d_self->d_epv->f_putBool))(d_self, /* in */ key.c_str(),
    /* in */ _local_value );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putIntArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<int> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putIntArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putIntArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putLongArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<long> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putLongArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putLongArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putFloatArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<float> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putFloatArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putFloatArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putDoubleArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<double> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putDoubleArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putDoubleArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putFcomplexArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::fcomplex> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putFcomplexArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putFcomplexArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putDcomplexArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::SIDL::dcomplex> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putDcomplexArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putDcomplexArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putStringArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array< ::std::string> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putStringArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putStringArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}


/**
 * user defined non-static method.
 */
void
framework::TypeMap::putBoolArray( /*in*/ const ::std::string& key,
  /*in*/ ::SIDL::array<bool> value )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::putBoolArray()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_putBoolArray))(d_self, /* in */ key.c_str(),
    /* in */ value._get_ior() );
  // unpack results and cleanup

}



/**
 * Make the key and associated value disappear from the object. 
 */
void
framework::TypeMap::remove( /*in*/ const ::std::string& key )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::remove()\""
    ));
  }

  // pack args to dispatch to ior

  // dispatch to ior
  (*(d_self->d_epv->f_remove))(d_self, /* in */ key.c_str() );
  // unpack results and cleanup

}



/**
 *  Get all the names associated with a particular type
 *  without exposing the data implementation details.  The keys
 *  will be returned in an arbitrary order. If type specified is
 *  None (no specification) all keys of all types are returned.
 */
::SIDL::array< ::std::string>
framework::TypeMap::getAllKeys( /*in*/ ::gov::cca::Type t )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::getAllKeys()\""
    ));
  }
  ::SIDL::array< ::std::string> _result;
  // pack args to dispatch to ior
  SIDL_string__array* _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_getAllKeys))(d_self,
    /* in */ (enum gov_cca_Type__enum)t );
  // unpack results and cleanup
  _result._set_ior(_local_result);
  return _result;
}



/**
 * Return true if the key exists in this map 
 */
bool
framework::TypeMap::hasKey( /*in*/ const ::std::string& key )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::hasKey()\""
    ));
  }
  bool _result;
  // pack args to dispatch to ior
  SIDL_bool _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_hasKey))(d_self, /* in */ key.c_str() );
  // unpack results and cleanup
  _result = (_local_result == TRUE);
  return _result;
}



/**
 * Return the type of the value associated with this key 
 */
::gov::cca::Type
framework::TypeMap::typeOf( /*in*/ const ::std::string& key )
throw ( ::SIDL::NullIORException ) 

{
  if ( d_self == 0 ) {
    throw ::SIDL::NullIORException( ::std::string (
      "Null IOR Pointer in \"framework::TypeMap::typeOf()\""
    ));
  }
  ::gov::cca::Type _result;
  // pack args to dispatch to ior
  enum gov_cca_Type__enum _local_result;
  // dispatch to ior
  _local_result = (*(d_self->d_epv->f_typeOf))(d_self, /* in */ key.c_str() );
  // unpack results and cleanup
  _result = (::gov::cca::Type)_local_result;
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::framework::TypeMap
framework::TypeMap::_create() {
  ::framework::TypeMap self( (*_get_ext()->createObject)() );
  // NOTE: reference count == 2. 
  //   (1 from createObject, 1 from IOR->C++)
  // Decrement this count back down to one.
  (*(self.d_self->d_epv->f_deleteReference))(self.d_self);
  return self;
}

// default destructor
framework::TypeMap::~TypeMap () {
  if ( d_self != 0 ) {
    deleteReference();
  }
}

// copy constructor
framework::TypeMap::TypeMap ( const ::framework::TypeMap& original ) {
  d_self = const_cast< ior_t*>(original.d_self);
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addReference();
  }
}

// assignment operator
::framework::TypeMap&
framework::TypeMap::operator=( const ::framework::TypeMap& rhs ) {
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
framework::TypeMap::TypeMap ( ::framework::TypeMap::ior_t* ior ) 
    : d_self( ior ), d_weak_reference(false) {
  if ( d_self != 0 ) {
    addReference();
  }
}

// Alternate constructor: does not call addReference()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
framework::TypeMap::TypeMap ( ::framework::TypeMap::ior_t* ior, bool isWeak ) 
    : d_self( ior ), d_weak_reference(isWeak) { 
}

// conversion from a StubBase
framework::TypeMap::TypeMap ( const ::SIDL::StubBase& base )
{
  d_self = reinterpret_cast< ior_t*>(base._cast("framework.TypeMap"));
  d_weak_reference = false;
  if (d_self != 0) {
    addReference();
  }
}

// protected method that implements casting
void* framework::TypeMap::_cast(const char* type) const
{
  void* ptr = 0;
  if ( d_self != 0 ) {
    ptr = reinterpret_cast< void*>((*d_self->d_epv->f__cast)(d_self, type));
  }
  return ptr;
}

// Static data type
const ::framework::TypeMap::ext_t * framework::TypeMap::s_ext;

// private static method to get static data type
const ::framework::TypeMap::ext_t *
framework::TypeMap::_get_ext()
  throw (::SIDL::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = framework_TypeMap__externals();
#else
    const ext_t *(*dll_f)(void) =
      (const ext_t *(*)(void)) ::SIDL::Loader::lookupSymbol(
        "framework_TypeMap__externals");
    s_ext = (dll_f ? (*dll_f)() : NULL);
    if (!s_ext) {
      throw ::SIDL::NullIORException( ::std::string (
        "cannot find implementation for framework.TypeMap; please set SIDL_DLL_PATH"
      ));
    }
#endif
  }
  return s_ext;
}

