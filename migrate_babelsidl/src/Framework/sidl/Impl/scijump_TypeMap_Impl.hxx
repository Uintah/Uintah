// 
// File:          scijump_TypeMap_Impl.hxx
// Symbol:        scijump.TypeMap-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_TypeMap_Impl_hxx
#define included_scijump_TypeMap_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_TypeMap_IOR_h
#include "scijump_TypeMap_IOR.h"
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
#ifndef included_scijump_TypeMap_hxx
#include "scijump_TypeMap.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.TypeMap._hincludes)

#include <string>
#include <map>
#include <list>

// Insert-Code-Here {scijump.TypeMap._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.TypeMap._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.TypeMap" (version 0.2.1)
   */
  class TypeMap_impl : public virtual ::scijump::TypeMap 
  // DO-NOT-DELETE splicer.begin(scijump.TypeMap._inherits)
  // Insert-Code-Here {scijump.TypeMap._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.TypeMap._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.TypeMap._implementation)

  private:
    template<class T>
    class TypeMapBase  {
    public:
      typedef typename std::map<std::string, T>::size_type size_type;

      TypeMapBase() {}
      ~TypeMapBase() {}
      TypeMapBase(const TypeMapBase<T>&);
      TypeMapBase<T>& operator=(const TypeMapBase<T>&);

      T get(const std::string& key, const T& dflt);
      void put(const std::string& key, const T& value);
      void getAllKeys( std::list<std::string>& list);
      void getAllKeys( ::sidl::array<std::string>& array) { return getAllKeys(array, 0); }
      void getAllKeys( ::sidl::array<std::string>& array, const int startIndex);
      bool hasKey(const std::string& key);
      bool remove(const std::string& key);
      size_type size() { return typeMap.size(); }

    private:
      typedef typename std::map<std::string, T>::iterator MapIter;
      typedef typename std::map<std::string, T>::const_iterator MapConstIter;
      std::map<std::string, T> typeMap;
    };

    TypeMapBase<int32_t> intMap;
    TypeMapBase<int64_t> longMap;
    TypeMapBase<float> floatMap;
    TypeMapBase<double> doubleMap;
    TypeMapBase<std::string> stringMap;
    TypeMapBase<bool> boolMap;
    TypeMapBase<std::complex<float> > fcomplexMap;
    TypeMapBase<std::complex<double> > dcomplexMap;
    TypeMapBase< ::sidl::array<int32_t> > intArrayMap;
    TypeMapBase< ::sidl::array<int64_t> > longArrayMap;
    TypeMapBase< ::sidl::array<float> > floatArrayMap;
    TypeMapBase< ::sidl::array<double> > doubleArrayMap;
    TypeMapBase< ::sidl::array<std::string> > stringArrayMap;
    TypeMapBase< ::sidl::array<bool> > boolArrayMap;
    TypeMapBase< ::sidl::array<std::complex<float> > > fcomplexArrayMap;
    TypeMapBase< ::sidl::array<std::complex<double> > > dcomplexArrayMap;

    // Insert-Code-Here {scijump.TypeMap._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(scijump.TypeMap._implementation)

  public:
    // default constructor, used for data wrapping(required)
    TypeMap_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      TypeMap_impl( struct scijump_TypeMap__object * ior ) : StubBase(ior,true),
        
    ::gov::cca::TypeMap((ior==NULL) ? NULL : &((*ior).d_gov_cca_typemap)) , 
      _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~TypeMap_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     *  Create an exact copy of this Map 
     */
    ::gov::cca::TypeMap
    cloneTypeMap_impl() ;

    /**
     *  Create a new Map with no key/value associations. 
     */
    ::gov::cca::TypeMap
    cloneEmpty_impl() ;
    /**
     * user defined non-static method.
     */
    int32_t
    getInt_impl (
      /* in */const ::std::string& key,
      /* in */int32_t dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    int64_t
    getLong_impl (
      /* in */const ::std::string& key,
      /* in */int64_t dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    float
    getFloat_impl (
      /* in */const ::std::string& key,
      /* in */float dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    double
    getDouble_impl (
      /* in */const ::std::string& key,
      /* in */double dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::std::complex<float>
    getFcomplex_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::complex<float>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::std::complex<double>
    getDcomplex_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::complex<double>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::std::string
    getString_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::string& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    bool
    getBool_impl (
      /* in */const ::std::string& key,
      /* in */bool dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<int32_t>
    getIntArray_impl (
      /* in */const ::std::string& key,
      /* in array<int> */::sidl::array<int32_t>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<int64_t>
    getLongArray_impl (
      /* in */const ::std::string& key,
      /* in array<long> */::sidl::array<int64_t>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<float>
    getFloatArray_impl (
      /* in */const ::std::string& key,
      /* in array<float> */::sidl::array<float>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<double>
    getDoubleArray_impl (
      /* in */const ::std::string& key,
      /* in array<double> */::sidl::array<double>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array< ::sidl::fcomplex>
    getFcomplexArray_impl (
      /* in */const ::std::string& key,
      /* in array<fcomplex> */::sidl::array< ::sidl::fcomplex>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array< ::sidl::dcomplex>
    getDcomplexArray_impl (
      /* in */const ::std::string& key,
      /* in array<dcomplex> */::sidl::array< ::sidl::dcomplex>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array< ::std::string>
    getStringArray_impl (
      /* in */const ::std::string& key,
      /* in array<string> */::sidl::array< ::std::string>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    ::sidl::array<bool>
    getBoolArray_impl (
      /* in */const ::std::string& key,
      /* in array<bool> */::sidl::array<bool>& dflt
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Assign a key and value. Any value previously assigned
     * to the same key will be overwritten so long as it
     * is of the same type. If types conflict, an exception occurs.
     */
    void
    putInt_impl (
      /* in */const ::std::string& key,
      /* in */int32_t value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putLong_impl (
      /* in */const ::std::string& key,
      /* in */int64_t value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putFloat_impl (
      /* in */const ::std::string& key,
      /* in */float value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putDouble_impl (
      /* in */const ::std::string& key,
      /* in */double value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putFcomplex_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::complex<float>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putDcomplex_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::complex<double>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putString_impl (
      /* in */const ::std::string& key,
      /* in */const ::std::string& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putBool_impl (
      /* in */const ::std::string& key,
      /* in */bool value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putIntArray_impl (
      /* in */const ::std::string& key,
      /* in array<int> */::sidl::array<int32_t>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putLongArray_impl (
      /* in */const ::std::string& key,
      /* in array<long> */::sidl::array<int64_t>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putFloatArray_impl (
      /* in */const ::std::string& key,
      /* in array<float> */::sidl::array<float>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putDoubleArray_impl (
      /* in */const ::std::string& key,
      /* in array<double> */::sidl::array<double>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putFcomplexArray_impl (
      /* in */const ::std::string& key,
      /* in array<fcomplex> */::sidl::array< ::sidl::fcomplex>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putDcomplexArray_impl (
      /* in */const ::std::string& key,
      /* in array<dcomplex> */::sidl::array< ::sidl::dcomplex>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putStringArray_impl (
      /* in */const ::std::string& key,
      /* in array<string> */::sidl::array< ::std::string>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;

    /**
     * user defined non-static method.
     */
    void
    putBoolArray_impl (
      /* in */const ::std::string& key,
      /* in array<bool> */::sidl::array<bool>& value
    )
    // throws:
    //     ::gov::cca::TypeMismatchException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Make the key and associated value disappear from the object. 
     */
    void
    remove_impl (
      /* in */const ::std::string& key
    )
    ;


    /**
     *  
     * Get all the names associated with a particular type
     * without exposing the data implementation details.  The keys
     * will be returned in an arbitrary order. If type specified is
     * NoType (no specification) all keys of all types are returned.
     */
    ::sidl::array< ::std::string>
    getAllKeys_impl (
      /* in */::gov::cca::Type t
    )
    ;


    /**
     *  Return true if the key exists in this map 
     */
    bool
    hasKey_impl (
      /* in */const ::std::string& key
    )
    ;


    /**
     *  Return the type of the value associated with this key 
     */
    ::gov::cca::Type
    typeOf_impl (
      /* in */const ::std::string& key
    )
    ;

  };  // end class TypeMap_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.TypeMap._hmisc)
// Insert-Code-Here {scijump.TypeMap._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.TypeMap._hmisc)

#endif
