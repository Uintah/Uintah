// 
// File:          framework_TypeMap_Impl.hxx
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_framework_TypeMap_Impl_hxx
#define included_framework_TypeMap_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_framework_TypeMap_IOR_h
#include "framework_TypeMap_IOR.h"
#endif
#ifndef included_framework_TypeMap_hxx
#include "framework_TypeMap.hxx"
#endif
#ifndef included_gov_cca_Type_hxx
#include "gov_cca_Type.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_gov_cca_TypeMismatchException_hxx
#include "gov_cca_TypeMismatchException.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.TypeMap._includes)
#include <Framework/TypeMap.h>
// DO-NOT-DELETE splicer.end(framework.TypeMap._includes)

namespace framework { 

  /**
   * Symbol "framework.TypeMap" (version 1.0)
   */
  class TypeMap_impl : public virtual UCXX ::framework::TypeMap 
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._inherits)
  // Insert-Code-Here {framework.TypeMap._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.TypeMap._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(framework.TypeMap._implementation)
    SCIRun::TypeMap* typeMap;
    // DO-NOT-DELETE splicer.end(framework.TypeMap._implementation)

  public:
    // default constructor, shouldn't be used (required)
    TypeMap_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      TypeMap_impl( struct framework_TypeMap__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~TypeMap_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:

      /**
       * user defined non-static method.
       */
      void
      setInternalData_impl (
        /* in */void* data
      )
      ;

      /**
       * user defined non-static method.
       */
      void*
      getInternalData_impl() ;

      /**
       * Create an exact copy of this Map 
       */
      UCXX ::gov::cca::TypeMap
      cloneTypeMap_impl() ;

      /**
       * Create a new Map with no key/value associations. 
       */
      UCXX ::gov::cca::TypeMap
      cloneEmpty_impl() ;
      /**
       * user defined non-static method.
       */
      int32_t
      getInt_impl (
        /* in */const ::std::string& key,
        /* in */int32_t dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      int64_t
      getLong_impl (
        /* in */const ::std::string& key,
        /* in */int64_t dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      float
      getFloat_impl (
        /* in */const ::std::string& key,
        /* in */float dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      double
      getDouble_impl (
        /* in */const ::std::string& key,
        /* in */double dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      ::std::complex<float>
      getFcomplex_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<float>& dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      ::std::complex<double>
      getDcomplex_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<double>& dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      ::std::string
      getString_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::string& dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      bool
      getBool_impl (
        /* in */const ::std::string& key,
        /* in */bool dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array<int32_t>
      getIntArray_impl (
        /* in */const ::std::string& key,
        /* in array<int> */UCXX ::sidl::array<int32_t> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array<int64_t>
      getLongArray_impl (
        /* in */const ::std::string& key,
        /* in array<long> */UCXX ::sidl::array<int64_t> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array<float>
      getFloatArray_impl (
        /* in */const ::std::string& key,
        /* in array<float> */UCXX ::sidl::array<float> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array<double>
      getDoubleArray_impl (
        /* in */const ::std::string& key,
        /* in array<double> */UCXX ::sidl::array<double> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array< UCXX ::sidl::fcomplex>
      getFcomplexArray_impl (
        /* in */const ::std::string& key,
        /* in array<fcomplex> */UCXX ::sidl::array< UCXX ::sidl::fcomplex> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array< UCXX ::sidl::dcomplex>
      getDcomplexArray_impl (
        /* in */const ::std::string& key,
        /* in array<dcomplex> */UCXX ::sidl::array< UCXX ::sidl::dcomplex> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array< ::std::string>
      getStringArray_impl (
        /* in */const ::std::string& key,
        /* in array<string> */UCXX ::sidl::array< ::std::string> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * user defined non-static method.
       */
      UCXX ::sidl::array<bool>
      getBoolArray_impl (
        /* in */const ::std::string& key,
        /* in array<bool> */UCXX ::sidl::array<bool> dflt
      )
      throw ( 
        UCXX ::gov::cca::TypeMismatchException, 
        UCXX ::sidl::RuntimeException
      );


      /**
       * Assign a key and value. Any value previously assigned
       * to the same key will be overwritten.
       */
      void
      putInt_impl (
        /* in */const ::std::string& key,
        /* in */int32_t value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putLong_impl (
        /* in */const ::std::string& key,
        /* in */int64_t value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putFloat_impl (
        /* in */const ::std::string& key,
        /* in */float value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putDouble_impl (
        /* in */const ::std::string& key,
        /* in */double value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putFcomplex_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<float>& value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putDcomplex_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::complex<double>& value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putString_impl (
        /* in */const ::std::string& key,
        /* in */const ::std::string& value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putBool_impl (
        /* in */const ::std::string& key,
        /* in */bool value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putIntArray_impl (
        /* in */const ::std::string& key,
        /* in array<int> */UCXX ::sidl::array<int32_t> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putLongArray_impl (
        /* in */const ::std::string& key,
        /* in array<long> */UCXX ::sidl::array<int64_t> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putFloatArray_impl (
        /* in */const ::std::string& key,
        /* in array<float> */UCXX ::sidl::array<float> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putDoubleArray_impl (
        /* in */const ::std::string& key,
        /* in array<double> */UCXX ::sidl::array<double> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putFcomplexArray_impl (
        /* in */const ::std::string& key,
        /* in array<fcomplex> */UCXX ::sidl::array< UCXX ::sidl::fcomplex> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putDcomplexArray_impl (
        /* in */const ::std::string& key,
        /* in array<dcomplex> */UCXX ::sidl::array< UCXX ::sidl::dcomplex> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putStringArray_impl (
        /* in */const ::std::string& key,
        /* in array<string> */UCXX ::sidl::array< ::std::string> value
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      putBoolArray_impl (
        /* in */const ::std::string& key,
        /* in array<bool> */UCXX ::sidl::array<bool> value
      )
      ;


      /**
       * Make the key and associated value disappear from the object. 
       */
      void
      remove_impl (
        /* in */const ::std::string& key
      )
      ;


      /**
       *  Get all the names associated with a particular type
       *  without exposing the data implementation details.  The keys
       *  will be returned in an arbitrary order. If type specified is
       *  None (no specification) all keys of all types are returned.
       */
      UCXX ::sidl::array< ::std::string>
      getAllKeys_impl (
        /* in */UCXX ::gov::cca::Type t
      )
      ;


      /**
       * Return true if the key exists in this map 
       */
      bool
      hasKey_impl (
        /* in */const ::std::string& key
      )
      ;


      /**
       * Return the type of the value associated with this key 
       */
      UCXX ::gov::cca::Type
      typeOf_impl (
        /* in */const ::std::string& key
      )
      ;

    };  // end class TypeMap_impl

  } // end namespace framework

  // DO-NOT-DELETE splicer.begin(framework.TypeMap._misc)
  // Insert-Code-Here {framework.TypeMap._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(framework.TypeMap._misc)

  #endif
