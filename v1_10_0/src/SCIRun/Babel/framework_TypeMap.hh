// 
// File:          framework_TypeMap.hh
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
#define included_framework_TypeMap_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace framework { 

  class TypeMap;
} // end namespace framework

// Some compilers need to define array template before the specializations
#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
namespace SIDL {
  template<>
  class array< ::framework::TypeMap>;
}

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
#ifndef included_gov_cca_Type_hh
#include "gov_cca_Type.hh"
#endif
#ifndef included_gov_cca_TypeMap_hh
#include "gov_cca_TypeMap.hh"
#endif
#ifndef included_gov_cca_TypeMismatchException_hh
#include "gov_cca_TypeMismatchException.hh"
#endif

namespace framework { 

  /**
   * Symbol "framework.TypeMap" (version 1.0)
   */
  class TypeMap : public ::SIDL::StubBase {

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

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
    addReference() throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     */
    void
    deleteReference() throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * Return true if and only if &amp;lt;code&amp;gt;obj&amp;lt;/code&amp;gt; refers to the same
     * object as this object.
     */
    bool
    isSame (
      /*in*/ ::SIDL::BaseInterface iobj
    )
    throw ( ::SIDL::NullIORException ) 
    ;



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
    queryInterface (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Return whether this object is an instance of the specified type.
     * The string name must be the &amp;lt;code&amp;gt;SIDL&amp;lt;/code&amp;gt; type name.  This
     * routine will return &amp;lt;code&amp;gt;true&amp;lt;/code&amp;gt; if and only if a cast to
     * the string type name would succeed.
     */
    bool
    isInstanceOf (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Create an exact copy of this Map 
     */
    ::gov::cca::TypeMap
    cloneTypeMap() throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * Create a new Map with no key/value associations. 
     */
    ::gov::cca::TypeMap
    cloneEmpty() throw ( ::SIDL::NullIORException ) 
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
      ::SIDL::NullIORException, ::gov::cca::TypeMismatchException
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
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putLong (
      /*in*/ const ::std::string& key,
      /*in*/ int64_t value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putFloat (
      /*in*/ const ::std::string& key,
      /*in*/ float value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putDouble (
      /*in*/ const ::std::string& key,
      /*in*/ double value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putFcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<float>& value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putDcomplex (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::complex<double>& value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putString (
      /*in*/ const ::std::string& key,
      /*in*/ const ::std::string& value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putBool (
      /*in*/ const ::std::string& key,
      /*in*/ bool value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putIntArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<int> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putLongArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<long> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putFloatArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<float> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putDoubleArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<double> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putFcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::fcomplex> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putDcomplexArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::SIDL::dcomplex> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putStringArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array< ::std::string> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    void
    putBoolArray (
      /*in*/ const ::std::string& key,
      /*in*/ ::SIDL::array<bool> value
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Make the key and associated value disappear from the object. 
     */
    void
    remove (
      /*in*/ const ::std::string& key
    )
    throw ( ::SIDL::NullIORException ) 
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
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Return true if the key exists in this map 
     */
    bool
    hasKey (
      /*in*/ const ::std::string& key
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Return the type of the value associated with this key 
     */
    ::gov::cca::Type
    typeOf (
      /*in*/ const ::std::string& key
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct framework_TypeMap__object ior_t;
    typedef struct framework_TypeMap__external ext_t;
    typedef struct framework_TypeMap__sepv sepv_t;

    // default constructor
    TypeMap() : d_self(0), d_weak_reference(false) { }

    // static constructor
    static ::framework::TypeMap _create();

    // default destructor
    virtual ~TypeMap ();

    // copy constructor
    TypeMap ( const TypeMap& original );

    // assignment operator
    TypeMap& operator= ( const TypeMap& rhs );

    // conversion from ior to C++ class
    TypeMap ( TypeMap::ior_t* ior );

    // Alternate constructor: does not call addReference()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    TypeMap ( TypeMap::ior_t* ior, bool isWeak );

    // conversion from a StubBase
    TypeMap ( const ::SIDL::StubBase& base );

    ior_t* _get_ior() { return d_self; }

    const ior_t* _get_ior() const { return d_self; }

    void _set_ior( ior_t* ptr ) { d_self = ptr; }

    bool _is_nil() const { return (d_self==0); }

    bool _not_nil() const { return (d_self!=0); }

    bool operator !() const { return (d_self==0); }

  protected:
    virtual void* _cast(const char* type) const;

  private:
    // Pointer to SIDL's IOR type (one per instance)
    ior_t * d_self;

    // Weak references (used by Impl's only) don't add/deleteRef()
    bool d_weak_reference;

    // Pointer to external (DLL loadable) symbols (shared among instances)
    static const ext_t * s_ext;

  public:
    static const ext_t * _get_ext() throw ( ::SIDL::NullIORException );

  }; // end class TypeMap
} // end namespace framework

// array specialization
namespace SIDL {
  template<>
  class array< ::framework::TypeMap> : public array_mixin< struct 
    framework_TypeMap__array, struct framework_TypeMap__object*,
    ::framework::TypeMap>{
  public:
    /**
     * default constructor
     */
    array() : array_mixin< struct framework_TypeMap__array,
      struct framework_TypeMap__object*, ::framework::TypeMap>() {}

    /**
     * default destructor
     */
    virtual ~array() { if ( d_array ) { deleteReference(); } }

    /**
     * copy constructor
     */
    array( const array< ::framework::TypeMap >& original ) {
      d_array = original.d_array;
      if ( d_array ) { addReference(); }
    }

    /**
     * assignment operator
     */
    array< ::framework::TypeMap >& operator=( const array< ::framework::TypeMap 
      >& rhs ) {
      if ( d_array != rhs.d_array ) {
        if ( d_array ) { deleteReference(); }
        d_array=rhs.d_array;
        if ( d_array ) { addReference(); }
      }
      return *this;
    }

  /**
   * conversion from ior to C++ class
   * (constructor/casting operator)
   */
  array( struct framework_TypeMap__array* src ) : array_mixin< struct 
    framework_TypeMap__array, struct framework_TypeMap__object*,
    ::framework::TypeMap>(src) {}

  /**
   * static constructor: createRow
   */
  static array < ::framework::TypeMap >
  createRow(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_createRow( dimen, lower, upper ) );
    return a;
  }

  /**
   * static constructor: createCol
   */
  static array < ::framework::TypeMap >
  createCol(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_createCol( dimen, lower, upper ) );
    return a;
  }

  /**
   * static constructor: create1d
   */
  static array < ::framework::TypeMap >
  create1d(int32_t len) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_create1d(len));
    return a;
  }

  /**
   * static constructor: create2dCol
   */
  static array < ::framework::TypeMap >
  create2dCol(int32_t m, int32_t n) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_create2dCol(m,n));
    return a;
  }

  /**
   * static constructor: create2dRow
   */
  static array < ::framework::TypeMap >
  create2dRow(int32_t m, int32_t n) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_create2dRow(m,n));
    return a;
  }

  /**
   * constructor: slice
   */
  array < ::framework::TypeMap >
  slice( int dimen,
         const int32_t newElem[],
         const int32_t *srcStart = 0,
         const int32_t *srcStride = 0,
         const int32_t *newStart = 0) {
    array < ::framework::TypeMap > a;
    a._set_ior( (struct framework_TypeMap__array*)
                SIDL_interface__array_slice( (struct SIDL_interface__array *) 
      d_array,
                                   dimen, newElem, srcStart, srcStride,
      newStart) );
    return a;
  }

  /**
   * copy
   */
  void copy( const array< ::framework::TypeMap>& src ) {
    SIDL_interface__array_copy( (const struct SIDL_interface__array*) 
      src._get_ior(),(struct SIDL_interface__array*) _get_ior() );
  }

  /**
   * constructor: smart copy
   */
  void smartCopy() {
    struct SIDL_interface__array * p  = SIDL_interface__array_smartCopy(
      (struct SIDL_interface__array *) _get_ior() );
    if ( _not_nil() ) { deleteReference(); }
    _set_ior( (struct framework_TypeMap__array*) p);
  }

  /**
   * constructor: ensure
   */
  void ensure(int32_t dimen, int ordering ) {
    struct SIDL_interface__array * p  = SIDL_interface__array_ensure(
      (struct SIDL_interface__array *) d_array, dimen, ordering );
    if ( _not_nil() ) { deleteReference(); }
    _set_ior( (struct framework_TypeMap__array*) p);
  }
};
}

#endif
