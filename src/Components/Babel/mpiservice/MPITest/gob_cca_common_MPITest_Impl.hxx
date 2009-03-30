// 
// File:          gob_cca_common_MPITest_Impl.hxx
// Symbol:        gob.cca.common.MPITest-v0.0
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for gob.cca.common.MPITest
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_gob_cca_common_MPITest_Impl_hxx
#define included_gob_cca_common_MPITest_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_gob_cca_common_MPITest_IOR_h
#include "gob_cca_common_MPITest_IOR.h"
#endif
#ifndef included_gob_cca_common_MPITest_hxx
#include "gob_cca_common_MPITest.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_ComponentRelease_hxx
#include "gov_cca_ComponentRelease.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_ports_GoPort_hxx
#include "gov_cca_ports_GoPort.hxx"
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


// DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._hincludes)
// Insert-Code-Here {gob.cca.common.MPITest._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._hincludes)

namespace gob { 
  namespace cca { 
    namespace common { 

      /**
       * Symbol "gob.cca.common.MPITest" (version 0.0)
       */
      class MPITest_impl : public virtual ::gob::cca::common::MPITest 
      // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._inherits)
      // Insert-Code-Here {gob.cca.common.MPITest._inherits} (optional inheritance here)
      // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._inherits)

      {

      // All data marked protected will be accessable by 
      // descendant Impl classes
      protected:

        bool _wrapped;

        // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._implementation)

  // Insert-UserCode-Here(gob.cca.common.MPITest._implementation)

  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest._implementation)
  
   gov::cca::Services    d_services; // our cca handle.
 

  // Bocca generated code. bocca.protected.end(gob.cca.common.MPITest._implementation)

        // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._implementation)

      public:
        // default constructor, used for data wrapping(required)
        MPITest_impl();
        // sidl constructor (required)
        // Note: alternate Skel constructor doesn't call addref()
        // (fixes bug #275)
          MPITest_impl( struct gob_cca_common_MPITest__object * ior ) : 
            StubBase(ior,true), 
          ::gov::cca::Component((ior==NULL) ? NULL : &((
            *ior).d_gov_cca_component)),
          ::gov::cca::ComponentRelease((ior==NULL) ? NULL : &((
            *ior).d_gov_cca_componentrelease)),
          ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
        ::gov::cca::ports::GoPort((ior==NULL) ? NULL : &((
          *ior).d_gov_cca_ports_goport)) , _wrapped(false) {_ctor();}


        // user defined construction
        void _ctor();

        // virtual destructor (required)
        virtual ~MPITest_impl() { _dtor(); }

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
        boccaSetServices_impl (
          /* in */::gov::cca::Services& services
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;

        /**
         * user defined non-static method.
         */
        void
        boccaReleaseServices_impl (
          /* in */::gov::cca::Services& services
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         *  Starts up a component presence in the calling framework.
         * @param services the component instance's handle on the framework world.
         * Contracts concerning Svc and setServices:
         * 
         * The component interaction with the CCA framework
         * and Ports begins on the call to setServices by the framework.
         * 
         * This function is called exactly once for each instance created
         * by the framework.
         * 
         * The argument Svc will never be nil/null.
         * 
         * Those uses ports which are automatically connected by the framework
         * (so-called service-ports) may be obtained via getPort during
         * setServices.
         */
        void
        setServices_impl (
          /* in */::gov::cca::Services& services
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         * Shuts down a component presence in the calling framework.
         * @param services the component instance's handle on the framework world.
         * Contracts concerning Svc and setServices:
         * 
         * This function is called exactly once for each callback registered
         * through Services.
         * 
         * The argument Svc will never be nil/null.
         * The argument Svc will always be the same as that received in
         * setServices.
         * 
         * During this call the component should release any interfaces
         * acquired by getPort().
         * 
         * During this call the component should reset to nil any stored
         * reference to Svc.
         * 
         * After this call, the component instance will be removed from the
         * framework. If the component instance was created by the
         * framework, it will be destroyed, not recycled, The behavior of
         * any port references obtained from this component instance and
         * stored elsewhere becomes undefined.
         * 
         * Notes for the component implementor:
         * 1) The component writer may perform blocking activities
         * within releaseServices, such as waiting for remote computations
         * to shutdown.
         * 2) It is good practice during releaseServices for the component
         * writer to remove or unregister all the ports it defined.
         */
        void
        releaseServices_impl (
          /* in */::gov::cca::Services& services
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         *  
         * Execute some encapsulated functionality on the component. 
         * Return 0 if ok, -1 if internal error but component may be 
         * used further, and -2 if error so severe that component cannot
         * be further used safely.
         */
        int32_t
        go_impl() ;
      };  // end class MPITest_impl

    } // end namespace common
  } // end namespace cca
} // end namespace gob

// DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._hmisc)
// Insert-Code-Here {gob.cca.common.MPITest._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._hmisc)

#endif
