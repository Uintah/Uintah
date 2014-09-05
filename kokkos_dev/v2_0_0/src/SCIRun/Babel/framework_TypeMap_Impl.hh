// 
// File:          framework_TypeMap_Impl.hh
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030720 10:32:36 MDT
// Generated:     20030720 10:32:38 MDT
// Description:   Server-side implementation for framework.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 17
// source-url    = file:/home/sci/kzhang/SCIRun/debug/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_TypeMap_Impl_hh
#define included_framework_TypeMap_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_TypeMap_IOR_h
#include "framework_TypeMap_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_TypeMap_hh
#include "framework_TypeMap.hh"
#endif
#ifndef included_gov_cca_Type_hh
#include "gov_cca_Type.hh"
#endif
#ifndef included_gov_cca_TypeMap_hh
#include "gov_cca_TypeMap.hh"
#endif
#ifndef included_gov_cca_TypeMismatchException_hh
#include "gov_cca_TypeMismatchException.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.TypeMap._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.TypeMap._includes)

namespace framework { 

  /**
   * Symbol "framework.TypeMap" (version 1.0)
   */
  class TypeMap_impl
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.TypeMap._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    TypeMap self;

    // DO-NOT-DELETE splicer.begin(framework.TypeMap._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.TypeMap._implementation)

  private:
    // private default constructor (required)
    TypeMap_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    TypeMap_impl( struct framework_TypeMap__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~TypeMap_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Create an exact copy of this Map 
     */
    ::gov::cca::TypeMap
    cloneTypeMap() throw () 
    ;

    /**
     * Create a new Map with no key/value associations. 
     */
    ::gov::cca::TypeMap
    cloneEmpty() throw () 
    ;
    /**
     * user defined non-static method.
     */
    int32_t
    getInt (
      /*in*/ const ::std::string& key,
      /*in*/ int32_t dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    int64_t
    getLong (
      /*in*/ const ::std::string& key,
      /*in*/ int64_t dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    float
    getFloat (
      /*in*/ const ::std::string& key,
      /*in*/ float dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    double
    getDouble (
      /*in*/ const ::std::string& key,
      /*in*/ double dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::std::complex<float>
    getFcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<float>& dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::std::complex<double>
    getDcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<double>& dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::std::string
    getString (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::string& dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    bool
    getBool (
      /*in*/ const ::std::string& key,
      /*in*/ bool dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array<int>
    getIntArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<int> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array<long>
    getLongArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<long> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array<float>
    getFloatArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<float> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array<double>
    getDoubleArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<double> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array< ::SIDL::fcomplex>
    getFcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::fcomplex> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array< ::SIDL::dcomplex>
    getDcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::dcomplex> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array< ::std::string>
    getStringArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::std::string> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );

    /**
     * user defined non-static method.
     */
    ::SIDL::array<bool>
    getBoolArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<bool> dflt
    )
    throw ( 
      ::gov::cca::TypeMismatchException
    );


    /**
     * Assign a key and value. Any value previously assigned
     * to the same key will be overwritten.  
     */
    void
    putInt (
      /*in*/ const ::std::string& key,
      /*in*/ int32_t value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putLong (
      /*in*/ const ::std::string& key,
      /*in*/ int64_t value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putFloat (
      /*in*/ const ::std::string& key,
      /*in*/ float value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putDouble (
      /*in*/ const ::std::string& key,
      /*in*/ double value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putFcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<float>& value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putDcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<double>& value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putString (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::string& value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putBool (
      /*in*/ const ::std::string& key,
      /*in*/ bool value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putIntArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<int> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putLongArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<long> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putFloatArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<float> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putDoubleArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<double> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putFcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::fcomplex> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putDcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::dcomplex> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putStringArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::std::string> value
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    void
    putBoolArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<bool> value
    )
    throw () 
    ;


    /**
     * Make the key and associated value disappear from the object. 
     */
    void
    remove (
      /*in*/ const ::std::string& key
    )
    throw () 
    ;


    /**
     *  Get all the names associated with a particular type
     *  without exposing the data implementation details.  The keys
     *  will be returned in an arbitrary order. If type specified is
     *  None (no specification) all keys of all types are returned.
     */
    ::SIDL::array< ::std::string>
    getAllKeys (
      /*in*/ ::gov::cca::Type t
    )
    throw () 
    ;


    /**
     * Return true if the key exists in this map 
     */
    bool
    hasKey (
      /*in*/ const ::std::string& key
    )
    throw () 
    ;


    /**
     * Return the type of the value associated with this key 
     */
    ::gov::cca::Type
    typeOf (
      /*in*/ const ::std::string& key
    )
    throw () 
    ;

  };  // end class TypeMap_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.TypeMap._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.TypeMap._misc)

#endif
