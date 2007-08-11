// 
// File:          scijump_core_ServiceInfo_Impl.hxx
// Symbol:        scijump.core.ServiceInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.core.ServiceInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_core_ServiceInfo_Impl_hxx
#define included_scijump_core_ServiceInfo_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_core_ServiceInfo_IOR_h
#include "scijump_core_ServiceInfo_IOR.h"
#endif
#ifndef included_sci_cca_core_NotInitializedException_hxx
#include "sci_cca_core_NotInitializedException.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_sci_cca_core_ServiceInfo_hxx
#include "sci_cca_core_ServiceInfo.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._hincludes)
// Insert-Code-Here {scijump.core.ServiceInfo._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._hincludes)

namespace scijump { 
  namespace core { 

    /**
     * Symbol "scijump.core.ServiceInfo" (version 0.2.1)
     */
    class ServiceInfo_impl : public virtual ::scijump::core::ServiceInfo 
    // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._inherits)
    // Insert-Code-Here {scijump.core.ServiceInfo._inherits} (optional inheritance here)
    // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._inherits)

    {

    // All data marked protected will be accessable by 
    // descendant Impl classes
    protected:

      bool _wrapped;

      // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._implementation)
      std::string serviceName;
      ::sci::cca::core::PortInfo servicePort;
      ::sci::cca::core::PortInfo requesterPort;
      // Insert-Code-Here {scijump.core.ServiceInfo._implementation} (additional details)
      // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._implementation)

    public:
      // default constructor, used for data wrapping(required)
      ServiceInfo_impl();
      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
        ServiceInfo_impl( struct scijump_core_ServiceInfo__object * ior ) : 
          StubBase(ior,true), 
      ::sci::cca::core::ServiceInfo((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_serviceinfo)) , _wrapped(false) {_ctor();}


      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~ServiceInfo_impl() { _dtor(); }

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
        /* in */const ::std::string& serviceName,
        /* in */::sci::cca::core::PortInfo& servicePort,
        /* in */::sci::cca::core::PortInfo& requesterPort
      )
      ;

      /**
       * user defined non-static method.
       */
      ::std::string
      getServiceName_impl() ;
      /**
       * user defined non-static method.
       */
      ::std::string
      getServicePortName_impl() ;
      /**
       * user defined non-static method.
       */
      ::sci::cca::core::PortInfo
      getServicePort_impl() // throws:
      //     ::sci::cca::core::NotInitializedException
      //     ::sidl::RuntimeException
      ;
      /**
       * user defined non-static method.
       */
      ::std::string
      getRequesterPortName_impl() ;
      /**
       * user defined non-static method.
       */
      ::sci::cca::core::PortInfo
      getRequesterPort_impl() // throws:
      //     ::sci::cca::core::NotInitializedException
      //     ::sidl::RuntimeException
      ;
    };  // end class ServiceInfo_impl

  } // end namespace core
} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._hmisc)
// Insert-Code-Here {scijump.core.ServiceInfo._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._hmisc)

#endif
