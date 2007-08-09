// 
// File:          scijump_Subscription_Impl.hxx
// Symbol:        scijump.Subscription-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.Subscription
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_Subscription_Impl_hxx
#define included_scijump_Subscription_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_Subscription_IOR_h
#include "scijump_Subscription_IOR.h"
#endif
#ifndef included_sci_cca_Event_hxx
#include "sci_cca_Event.hxx"
#endif
#ifndef included_sci_cca_EventListener_hxx
#include "sci_cca_EventListener.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_sci_cca_Subscription_hxx
#include "sci_cca_Subscription.hxx"
#endif
#ifndef included_scijump_SCIJumpFramework_hxx
#include "scijump_SCIJumpFramework.hxx"
#endif
#ifndef included_scijump_Subscription_hxx
#include "scijump_Subscription.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.Subscription._hincludes)
#include <map>
#include <scijump_EventListener.hxx>
// DO-NOT-DELETE splicer.end(scijump.Subscription._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.Subscription" (version 0.2.1)
   */
  class Subscription_impl : public virtual ::scijump::Subscription 
  // DO-NOT-DELETE splicer.begin(scijump.Subscription._inherits)
  // Insert-Code-Here {scijump.Subscription._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.Subscription._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.Subscription._implementation)
    std::string subscriptionName;
    SCIJumpFramework sjf;

    typedef std::map<std::string, scijump::EventListener> EventListenerMap;
    EventListenerMap eventListenerMap;
    // DO-NOT-DELETE splicer.end(scijump.Subscription._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Subscription_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Subscription_impl( struct scijump_Subscription__object * ior ) : StubBase(
        ior,true), 
    ::sci::cca::Subscription((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_subscription)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Subscription_impl() { _dtor(); }

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
      /* in */const ::std::string& subscriptionName,
      /* in */::scijump::SCIJumpFramework& sjf
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    processEvents_impl (
      /* in array<sci.cca.Event> */::sidl::array< ::sci::cca::Event>& eventList
    )
    ;


    /**
     *  Adds a listener to the collection of listeners for this Subscription.
     * 
     * @listenerKey - It is used as an index to the collection (STL
     * map) and the parameter \em theListener is a
     * pointer to the /em Listener class.
     * @theListener - A pointer to the object that will listen for events.
     */
    void
    registerEventListener_impl (
      /* in */const ::std::string& listenerKey,
      /* in */::sci::cca::EventListener& theListener
    )
    // throws:
    //     ::sci::cca::EventServiceException
    //     ::sidl::RuntimeException
    ;


    /**
     * Removes a listener from the collection of listeners for this Topic.
     * 
     * @listenerKey - It is used as an index to remove this listener.
     */
    void
    unregisterEventListener_impl (
      /* in */const ::std::string& listenerKey
    )
    ;


    /**
     *  Returns the name for this Subscription object 
     */
    ::std::string
    getSubscriptionName_impl() ;
  };  // end class Subscription_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.Subscription._hmisc)
// Insert-Code-Here {scijump.Subscription._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.Subscription._hmisc)

#endif
