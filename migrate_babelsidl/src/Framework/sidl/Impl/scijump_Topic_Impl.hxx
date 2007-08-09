// 
// File:          scijump_Topic_Impl.hxx
// Symbol:        scijump.Topic-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
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
#ifndef included_scijump_SCIJumpFramework_hxx
#include "scijump_SCIJumpFramework.hxx"
#endif
#ifndef included_scijump_Subscription_hxx
#include "scijump_Subscription.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.Topic._hincludes)
#include <map>
#include <string>
#include <vector>
#include <scijump_Event.hxx>
#include <scijump_Subscription.hxx>
// DO-NOT-DELETE splicer.end(scijump.Topic._hincludes)

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
    std::vector< scijump::Event> eventList;

    typedef std::map< std::string, Subscription> SubscriptionMap;
    SubscriptionMap subscriptionMap;

    SCIJumpFramework sjf;
    // DO-NOT-DELETE splicer.end(scijump.Topic._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Topic_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Topic_impl( struct scijump_Topic__object * ior ) : StubBase(ior,true), 
    ::sci::cca::Topic((ior==NULL) ? NULL : &((*ior).d_sci_cca_topic)) , 
      _wrapped(false) {_ctor();}


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
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */const ::std::string& topicName,
      /* in */::scijump::SCIJumpFramework& sjf
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    addSubscription_impl (
      /* in */const ::std::string& topicName,
      /* in */::scijump::Subscription& theSubscription
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    removeSubscription_impl (
      /* in */const ::std::string& topicName
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    processEvents_impl() ;

    /**
     *  Returns the topic name associated with this object 
     */
    ::std::string
    getTopicName_impl() ;

    /**
     *  Publish an event. 
     * 
     * @eventName - The name of this event. It is perhaps not a crucial
     * piece of information. Can be inserted into the
     * header or the body of the event by the event
     * service.
     * @eventBody - A typemap containing all the information to be 
     * sent out.
     */
    void
    sendEvent_impl (
      /* in */const ::std::string& eventName,
      /* in */::gov::cca::TypeMap& eventBody
    )
    // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;

  };  // end class Topic_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.Topic._hmisc)
// Insert-Code-Here {scijump.Topic._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.Topic._hmisc)

#endif
