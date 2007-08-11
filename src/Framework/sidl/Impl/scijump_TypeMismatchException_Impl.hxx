// 
// File:          scijump_TypeMismatchException_Impl.hxx
// Symbol:        scijump.TypeMismatchException-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.TypeMismatchException
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_TypeMismatchException_Impl_hxx
#define included_scijump_TypeMismatchException_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_TypeMismatchException_IOR_h
#include "scijump_TypeMismatchException_IOR.h"
#endif
#ifndef included_gov_cca_CCAExceptionType_hxx
#include "gov_cca_CCAExceptionType.hxx"
#endif
#ifndef included_gov_cca_Type_hxx
#include "gov_cca_Type.hxx"
#endif
#ifndef included_gov_cca_TypeMismatchException_hxx
#include "gov_cca_TypeMismatchException.hxx"
#endif
#ifndef included_scijump_TypeMismatchException_hxx
#include "scijump_TypeMismatchException.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_SIDLException_hxx
#include "sidl_SIDLException.hxx"
#endif
#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
#endif
#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
#endif


// DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._hincludes)
// Insert-Code-Here {scijump.TypeMismatchException._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.TypeMismatchException" (version 0.2.1)
   */
  class TypeMismatchException_impl : public virtual 
    ::scijump::TypeMismatchException 
  // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._inherits)
  // Insert-Code-Here {scijump.TypeMismatchException._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._implementation)
    ::gov::cca::CCAExceptionType type;
    ::gov::cca::Type requestedType;
    ::gov::cca::Type actualType;
    // DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._implementation)

  public:
    // default constructor, used for data wrapping(required)
    TypeMismatchException_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      TypeMismatchException_impl( struct scijump_TypeMismatchException__object 
        * ior ) : StubBase(ior,true), 
      ::sidl::io::Serializable((ior==NULL) ? NULL : &((
        *ior).d_sidl_sidlexception.d_sidl_io_serializable)),
      ::sidl::BaseException((ior==NULL) ? NULL : &((
        *ior).d_sidl_sidlexception.d_sidl_baseexception)),
      ::gov::cca::CCAException((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_ccaexception)),
    ::gov::cca::TypeMismatchException((ior==NULL) ? NULL : &((
      *ior).d_gov_cca_typemismatchexception)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~TypeMismatchException_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */::gov::cca::Type requestedType,
      /* in */::gov::cca::Type actualType
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */::gov::cca::CCAExceptionType type,
      /* in */::gov::cca::Type requestedType,
      /* in */::gov::cca::Type actualType
    )
    ;

    /**
     * user defined non-static method.
     */
    ::gov::cca::CCAExceptionType
    getCCAExceptionType_impl() ;

    /**
     *  @return the enumerated value Type sought 
     */
    ::gov::cca::Type
    getRequestedType_impl() ;

    /**
     *  @return the enumerated value Type sought 
     */
    ::gov::cca::Type
    getActualType_impl() ;
  };  // end class TypeMismatchException_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.TypeMismatchException._hmisc)
// Insert-Code-Here {scijump.TypeMismatchException._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.TypeMismatchException._hmisc)

#endif
