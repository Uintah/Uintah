// 
// File:          scijump_EventService_Impl.hxx
// Symbol:        scijump.EventService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.EventService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_EventService_Impl_hxx
#define included_scijump_EventService_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_EventService_IOR_h
#include "scijump_EventService_IOR.h"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_sci_cca_Subscription_hxx
#include "sci_cca_Subscription.hxx"
#endif
#ifndef included_sci_cca_Topic_hxx
#include "sci_cca_Topic.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sci_cca_ports_PublisherEventService_hxx
#include "sci_cca_ports_PublisherEventService.hxx"
#endif
#ifndef included_sci_cca_ports_SubscriberEventService_hxx
#include "sci_cca_ports_SubscriberEventService.hxx"
#endif
#ifndef included_scijump_EventService_hxx
#include "scijump_EventService.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.EventService._hincludes)

#include <map>

#include "scijump_Subscription.hxx"
#include "scijump_Topic.hxx"

// DO-NOT-DELETE splicer.end(scijump.EventService._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.EventService" (version 0.2.1)
   */
  class EventService_impl : public virtual ::scijump::EventService 
  // DO-NOT-DELETE splicer.begin(scijump.EventService._inherits)
  // Insert-Code-Here {scijump.EventService._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.EventService._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.EventService._implementation)
    typedef std::map<std::string, Subscription> SubscriptionMap;
    typedef std::map<std::string, Topic> TopicMap;

    TopicMap topicMap;
    SubscriptionMap subscriptionMap;
    ::sci::cca::AbstractFramework framework;

    bool isMatch(const std::string& topicName, 
		 const std::string& subscriptionName);
    // DO-NOT-DELETE splicer.end(scijump.EventService._implementation)

  public:
    // default constructor, used for data wrapping(required)
    EventService_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      EventService_impl( struct scijump_EventService__object * ior ) : StubBase(
        ior,true), 
      ::sci::cca::core::FrameworkService((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_frameworkservice)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
      ::sci::cca::ports::PublisherEventService((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_ports_publishereventservice)),
    ::sci::cca::ports::SubscriberEventService((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_ports_subscribereventservice)) , _wrapped(false) {_ctor(
      );}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~EventService_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:
    /**
     * user defined static method
     */
    static ::sci::cca::core::FrameworkService
    create_impl (
      /* in */::sci::cca::AbstractFramework& framework
    )
    ;


    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */::sci::cca::AbstractFramework& framework
    )
    ;


    /**
     *  Get a Topic by passing a name that has the form X.Y.Z. The
     * method creates a topic of topicName it if it doesn't exist.
     * 
     * @topicName - A dot delimited, hierarchical name of the topic
     * on which to publish. Wildcard characters are not
     * allowed for a topicName.
     */
    ::sci::cca::Topic
    getTopic_impl (
      /* in */const ::std::string& topicName
    )
    // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Returns true if topic already exists, false otherwise 
     */
    bool
    existsTopic_impl (
      /* in */const ::std::string& topicName
    )
    ;


    /**
     *  Subscribe to one or more topics.
     * 
     * @subscriptionName - A dot delimited hierarchical name selecting
     * the list of topics to get events from. Wildcard
     * characters (,?)  are allowed for a subscriptionName
     * to denote more than one topic.
     */
    ::sci::cca::Subscription
    getSubscription_impl (
      /* in */const ::std::string& subscriptionName
    )
    // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Process published events. When the subscriber calls this method,
     * this thread or some other one delivers each event by calling
     * processEvent(...) on each listener belonging to each specific
     * Subscription 
     */
    void
    processEvents_impl() // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;
  };  // end class EventService_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.EventService._hmisc)
// Insert-Code-Here {scijump.EventService._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.EventService._hmisc)

#endif
