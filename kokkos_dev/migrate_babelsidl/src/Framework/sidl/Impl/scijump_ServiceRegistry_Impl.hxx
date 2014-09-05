// 
// File:          scijump_ServiceRegistry_Impl.hxx
// Symbol:        scijump.ServiceRegistry-v0.2.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for scijump.ServiceRegistry
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_ServiceRegistry_Impl_hxx
#define included_scijump_ServiceRegistry_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_ServiceRegistry_IOR_h
#include "scijump_ServiceRegistry_IOR.h"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
#endif
#ifndef included_gov_cca_ports_ServiceProvider_hxx
#include "gov_cca_ports_ServiceProvider.hxx"
#endif
#ifndef included_gov_cca_ports_ServiceRegistry_hxx
#include "gov_cca_ports_ServiceRegistry.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_ServiceRegistry_hxx
#include "scijump_ServiceRegistry.hxx"
#endif
#ifndef included_scijump_core_ServiceInfo_hxx
#include "scijump_core_ServiceInfo.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._hincludes)
#include <map>
#include <scijump_BabelPortInfo.hxx>
// DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.ServiceRegistry" (version 0.2.1)
   */
  class ServiceRegistry_impl : public virtual ::scijump::ServiceRegistry 
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._inherits)
  // Insert-Code-Here {scijump.ServiceRegistry._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._implementation)
    sci::cca::AbstractFramework framework;

    typedef std::map<std::string, scijump::BabelPortInfo> portMap;
    portMap singletons;
    // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._implementation)

  public:
    // default constructor, used for data wrapping(required)
    ServiceRegistry_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      ServiceRegistry_impl( struct scijump_ServiceRegistry__object * ior ) : 
        StubBase(ior,true), 
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
      ::gov::cca::ports::ServiceRegistry((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_ports_serviceregistry)),
    ::sci::cca::core::FrameworkService((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_core_frameworkservice)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~ServiceRegistry_impl() { _dtor(); }

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
     * user defined non-static method.
     */
    ::scijump::core::ServiceInfo
    getService_impl (
      /* in */const ::std::string& serviceName,
      /* in */::sci::cca::core::PortInfo& requesterPort
    )
    ;


    /**
     * Add a ServiceProvider that can be asked to produce service Port's
     * for other components to use subsequently.
     * True means success. False means that for some reason, the
     * provider isn't going to function. Possibly another server is doing
     * the job.
     */
    bool
    addService_impl (
      /* in */const ::std::string& serviceType,
      /* in */::gov::cca::ports::ServiceProvider& portProvider
    )
    // throws:
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;


    /**
     *  Add a "reusable" service gov.cca.Port for other components to use 
     * subsequently.
     */
    bool
    addSingletonService_impl (
      /* in */const ::std::string& serviceType,
      /* in */::gov::cca::Port& server
    )
    // throws:
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;


    /**
     *  Inform the framework that this service Port is no longer to
     * be used, subsequent to this call. 
     */
    void
    removeService_impl (
      /* in */const ::std::string& serviceType
    )
    // throws:
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;

  };  // end class ServiceRegistry_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._hmisc)
// Insert-Code-Here {scijump.ServiceRegistry._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._hmisc)

#endif
