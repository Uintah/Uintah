// 
// File:          framework_Services.hh
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20021109 17:19:36 MST
// Generated:     20021109 17:19:39 MST
// Description:   Client-side glue code for framework.Services
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sparker/SCIRun/cca/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_Services_hh
#define included_framework_Services_hh

// declare class before #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace framework { 

  class Services;
} // end namespace framework

// Some compilers need to define array template before the specializations
#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
namespace SIDL {
  template<>
  class array< ::framework::Services>;
}

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_Services_IOR_h
#include "framework_Services_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_govcca_ComponentID_hh
#include "govcca_ComponentID.hh"
#endif
#ifndef included_govcca_Port_hh
#include "govcca_Port.hh"
#endif
#ifndef included_govcca_TypeMap_hh
#include "govcca_TypeMap.hh"
#endif

namespace framework { 

  /**
   * Symbol "framework.Services" (version 1.0)
   */
  class Services : public ::SIDL::StubBase {

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
     * user defined non-static method.
     */
    void*
    getData() throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * Ask for a previously registered Port; will return a Port or generate an error. 
     */
    ::govcca::Port
    getPort (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Ask for a previously registered Port and return that Port if it is
     * available or return null otherwise. 
     */
    ::govcca::Port
    getPortNonblocking (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Modified according to Motion 31 
     */
    void
    registerUsesPort (
      /*in*/ const ::std::string& name,
      /*in*/ const ::std::string& type,
      /*in*/ ::govcca::TypeMap properties
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Notify the framework that a Port, previously registered by this component,
     * is no longer needed. 
     */
    void
    unregisterUsesPort (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Exports a Port implemented by this component to the framework.  
     * This Port is now available for the framework to connect to other components. 
     * Modified according to Motion 31 
     */
    void
    addProvidesPort (
      /*in*/ ::govcca::Port inPort,
      /*in*/ const ::std::string& name,
      /*in*/ const ::std::string& type,
      /*in*/ ::govcca::TypeMap properties
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Notifies the framework that a previously exported Port is no longer 
     * available for use.
     */
    void
    removeProvidesPort (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Notifies the framework that this component is finished with this Port.   
     * releasePort() method calls exactly match getPort() mehtod calls.  After 
     * releasePort() is invoked all references to the released Port become invalid. 
     */
    void
    releasePort (
      /*in*/ const ::std::string& name
    )
    throw ( ::SIDL::NullIORException ) 
    ;


    /**
     * user defined non-static method.
     */
    ::govcca::TypeMap
    createTypeMap() throw ( ::SIDL::NullIORException ) 
    ;

    /**
     * user defined non-static method.
     */
    ::govcca::TypeMap
    getPortProperties (
      /*in*/ const ::std::string& portName
    )
    throw ( ::SIDL::NullIORException ) 
    ;



    /**
     * Get a reference to the component to which this Services object belongs. 
     */
    ::govcca::ComponentID
    getComponentID() throw ( ::SIDL::NullIORException ) 
    ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct framework_Services__object ior_t;
    typedef struct framework_Services__external ext_t;
    typedef struct framework_Services__sepv sepv_t;

    // default constructor
    Services() : d_self(0), d_weak_reference(false) { }

    // static constructor
    static ::framework::Services _create();

    // default destructor
    virtual ~Services ();

    // copy constructor
    Services ( const Services& original );

    // assignment operator
    Services& operator= ( const Services& rhs );

    // conversion from ior to C++ class
    Services ( Services::ior_t* ior );

    // Alternate constructor: does not call addReference()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    Services ( Services::ior_t* ior, bool isWeak );

    // conversion from a StubBase
    Services ( const ::SIDL::StubBase& base );

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

  }; // end class Services
} // end namespace framework

// array specialization
namespace SIDL {
  template<>
  class array< ::framework::Services> : public array_mixin< struct 
    framework_Services__array, struct framework_Services__object*,
    ::framework::Services>{
  public:
    /**
     * default constructor
     */
    array() : array_mixin< struct framework_Services__array,
      struct framework_Services__object*, ::framework::Services>() {}

    /**
     * default destructor
     */
    virtual ~array() { if ( d_array ) { deleteReference(); } }

    /**
     * copy constructor
     */
    array( const array< ::framework::Services >& original ) {
      d_array = original.d_array;
      if ( d_array ) { addReference(); }
    }

    /**
     * assignment operator
     */
    array< ::framework::Services >& operator=( const array< 
      ::framework::Services >& rhs ) {
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
  array( struct framework_Services__array* src ) : array_mixin< struct 
    framework_Services__array, struct framework_Services__object*,
    ::framework::Services>(src) {}

  /**
   * static constructor: createRow
   */
  static array < ::framework::Services >
  createRow(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_createRow( dimen, lower, upper ) );
    return a;
  }

  /**
   * static constructor: createCol
   */
  static array < ::framework::Services >
  createCol(int32_t dimen, const int32_t lower[], const int32_t upper[]) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_createCol( dimen, lower, upper ) );
    return a;
  }

  /**
   * static constructor: create1d
   */
  static array < ::framework::Services >
  create1d(int32_t len) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_create1d(len));
    return a;
  }

  /**
   * static constructor: create2dCol
   */
  static array < ::framework::Services >
  create2dCol(int32_t m, int32_t n) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_create2dCol(m,n));
    return a;
  }

  /**
   * static constructor: create2dRow
   */
  static array < ::framework::Services >
  create2dRow(int32_t m, int32_t n) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_create2dRow(m,n));
    return a;
  }

  /**
   * constructor: slice
   */
  array < ::framework::Services >
  slice( int dimen,
         const int32_t newElem[],
         const int32_t *srcStart = 0,
         const int32_t *srcStride = 0,
         const int32_t *newStart = 0) {
    array < ::framework::Services > a;
    a._set_ior( (struct framework_Services__array*)
                SIDL_interface__array_slice( (struct SIDL_interface__array *) 
      d_array,
                                   dimen, newElem, srcStart, srcStride,
      newStart) );
    return a;
  }

  /**
   * copy
   */
  void copy( const array< ::framework::Services>& src ) {
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
    _set_ior( (struct framework_Services__array*) p);
  }

  /**
   * constructor: ensure
   */
  void ensure(int32_t dimen, int ordering ) {
    struct SIDL_interface__array * p  = SIDL_interface__array_ensure(
      (struct SIDL_interface__array *) d_array, dimen, ordering );
    if ( _not_nil() ) { deleteReference(); }
    _set_ior( (struct framework_Services__array*) p);
  }
};
}

#endif
