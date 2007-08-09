// 
// File:          scijump_Event_Impl.hxx
// Symbol:        scijump.Event-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.Event
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_Event_Impl_hxx
#define included_scijump_Event_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_Event_IOR_h
#include "scijump_Event_IOR.h"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_Event_hxx
#include "sci_cca_Event.hxx"
#endif
#ifndef included_scijump_Event_hxx
#include "scijump_Event.hxx"
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
#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
#endif
#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
#endif


// DO-NOT-DELETE splicer.begin(scijump.Event._hincludes)
// Insert-Code-Here {scijump.Event._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.Event._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.Event" (version 0.2.1)
   */
  class Event_impl : public virtual ::scijump::Event 
  // DO-NOT-DELETE splicer.begin(scijump.Event._inherits)
  // Insert-Code-Here {scijump.Event._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.Event._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.Event._implementation)
    // Insert-Code-Here {scijump.Event._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(scijump.Event._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Event_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Event_impl( struct scijump_Event__object * ior ) : StubBase(ior,true), 
      ::sidl::io::Serializable((ior==NULL) ? NULL : &((
        *ior).d_sidl_io_serializable)),
    ::sci::cca::Event((ior==NULL) ? NULL : &((*ior).d_sci_cca_event)) , 
      _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Event_impl() { _dtor(); }

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
    setHeader_impl (
      /* in */::gov::cca::TypeMap& h
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setBody_impl (
      /* in */::gov::cca::TypeMap& b
    )
    ;


    /**
     *  Return the event's header. The header is usually generated
     * by the framework and holds bookkeeping information
     */
    ::gov::cca::TypeMap
    getHeader_impl() ;

    /**
     *  Returs the event's body. The body is the information the 
     * publisher is sending to the subscribers
     */
    ::gov::cca::TypeMap
    getBody_impl() ;
    /**
     * user defined non-static method.
     */
    void
    packObj_impl (
      /* in */::sidl::io::Serializer& ser
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    unpackObj_impl (
      /* in */::sidl::io::Deserializer& des
    )
    ;

  };  // end class Event_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.Event._hmisc)
// Insert-Code-Here {scijump.Event._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.Event._hmisc)

#endif
