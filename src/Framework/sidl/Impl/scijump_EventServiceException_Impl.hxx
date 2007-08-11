// 
// File:          scijump_EventServiceException_Impl.hxx
// Symbol:        scijump.EventServiceException-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.EventServiceException
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_EventServiceException_Impl_hxx
#define included_scijump_EventServiceException_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_EventServiceException_IOR_h
#include "scijump_EventServiceException_IOR.h"
#endif
#ifndef included_gov_cca_CCAExceptionType_hxx
#include "gov_cca_CCAExceptionType.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_scijump_EventServiceException_hxx
#include "scijump_EventServiceException.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.EventServiceException._hincludes)
// Insert-Code-Here {scijump.EventServiceException._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.EventServiceException._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.EventServiceException" (version 0.2.1)
   */
  class EventServiceException_impl : public virtual 
    ::scijump::EventServiceException 
  // DO-NOT-DELETE splicer.begin(scijump.EventServiceException._inherits)
  // Insert-Code-Here {scijump.EventServiceException._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.EventServiceException._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.EventServiceException._implementation)
    ::gov::cca::CCAExceptionType type;
    // DO-NOT-DELETE splicer.end(scijump.EventServiceException._implementation)

  public:
    // default constructor, used for data wrapping(required)
    EventServiceException_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      EventServiceException_impl( struct scijump_EventServiceException__object 
        * ior ) : StubBase(ior,true), 
      ::sidl::io::Serializable((ior==NULL) ? NULL : &((
        *ior).d_sidl_sidlexception.d_sidl_io_serializable)),
      ::sidl::BaseException((ior==NULL) ? NULL : &((
        *ior).d_sidl_sidlexception.d_sidl_baseexception)),
      ::gov::cca::CCAException((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_ccaexception)),
    ::sci::cca::EventServiceException((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_eventserviceexception)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~EventServiceException_impl() { _dtor(); }

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
      /* in */::gov::cca::CCAExceptionType type
    )
    ;

    /**
     * user defined non-static method.
     */
    ::gov::cca::CCAExceptionType
    getCCAExceptionType_impl() ;
  };  // end class EventServiceException_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.EventServiceException._hmisc)
// Insert-Code-Here {scijump.EventServiceException._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.EventServiceException._hmisc)

#endif
