// 
// File:          scijump_Topic_Impl.hxx
// Symbol:        scijump.Topic-v0.2.1
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for scijump.Topic
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_Topic_Impl_hxx
#define included_scijump_Topic_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_Topic_IOR_h
#include "scijump_Topic_IOR.h"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_sci_cca_Topic_hxx
#include "sci_cca_Topic.hxx"
#endif
#ifndef included_scijump_Topic_hxx
#include "scijump_Topic.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.Topic._includes)
#include <string>;
// DO-NOT-DELETE splicer.end(scijump.Topic._includes)

namespace scijump { 

  /**
   * Symbol "scijump.Topic" (version 0.2.1)
   */
  class Topic_impl : public virtual ::scijump::Topic 
  // DO-NOT-DELETE splicer.begin(scijump.Topic._inherits)
  // Insert-Code-Here {scijump.Topic._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.Topic._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.Topic._implementation)
    std::string topicName;
    // DO-NOT-DELETE splicer.end(scijump.Topic._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Topic_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Topic_impl( struct scijump_Topic__object * s ) : StubBase(s,true), _wrapped(
      false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Topic_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     * @return Topic name (from EventService.createTopic)
     */
    ::std::string
    getTopicName_impl() ;

    /**
     *  Publish an event. 
     */
    void
    sendEvent_impl (
      /* in */const ::std::string& name,
      /* in */::gov::cca::TypeMap eventBody
    )
    // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;

  };  // end class Topic_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.Topic._misc)
// Insert-Code-Here {scijump.Topic._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.Topic._misc)

#endif
