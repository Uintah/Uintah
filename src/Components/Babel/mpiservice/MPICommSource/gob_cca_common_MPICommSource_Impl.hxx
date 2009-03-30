// 
// File:          gob_cca_common_MPICommSource_Impl.hxx
// Symbol:        gob.cca.common.MPICommSource-v0.0
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for gob.cca.common.MPICommSource
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_gob_cca_common_MPICommSource_Impl_hxx
#define included_gob_cca_common_MPICommSource_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_gob_cca_common_MPICommSource_IOR_h
#include "gob_cca_common_MPICommSource_IOR.h"
#endif
#ifndef included_gob_cca_common_MPICommSource_hxx
#include "gob_cca_common_MPICommSource.hxx"
#endif
#ifndef included_gob_cca_ports_MPIService_hxx
#include "gob_cca_ports_MPIService.hxx"
#endif
#ifndef included_gob_cca_ports_MPISetup_hxx
#include "gob_cca_ports_MPISetup.hxx"
#endif
#ifndef included_gov_cca_AbstractFramework_hxx
#include "gov_cca_AbstractFramework.hxx"
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


// DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._hincludes)

#include <gov_cca_ports_ServiceRegistry.hxx>

namespace priv_gob_cca_common {
	/** a class to keep mpi out of the impl header */
        class MPIService_Impl;
}

// DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._hincludes)

namespace gob { 
  namespace cca { 
    namespace common { 

      /**
       * Symbol "gob.cca.common.MPICommSource" (version 0.0)
       */
      class MPICommSource_impl : public virtual 
        ::gob::cca::common::MPICommSource 
      // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._inherits)
      // Insert-Code-Here {gob.cca.common.MPICommSource._inherits} (optional inheritance here)
      // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._inherits)

      {

      // All data marked protected will be accessable by 
      // descendant Impl classes
      protected:

        bool _wrapped;

        // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._implementation)

        gov::cca::AbstractFramework naf;
        gov::cca::ports::ServiceRegistry sr;
        gob::cca::ports::MPIService mpis;
        priv_gob_cca_common::MPIService_Impl *m1;
        /** true if component has been started already. */
        bool initialized;
        /** true if component has been shut down already. */
        bool finalized;
        /** true if component is initialized rather than loaded as a normal component. */
        bool isExternal;

        /* d_services must already be defined when this is called
         * and after its done mpis, m1,m2 will have their
         * best available values.
         */
        void initPrivate(int64_t dupComm, bool registery)
        throw (
                ::gov::cca::CCAException
        );


  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPICommSource._implementation)
  
   gov::cca::Services    d_services; // our cca handle.
 

  // Bocca generated code. bocca.protected.end(gob.cca.common.MPICommSource._implementation)

        // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._implementation)

      public:
        // default constructor, used for data wrapping(required)
        MPICommSource_impl();
        // sidl constructor (required)
        // Note: alternate Skel constructor doesn't call addref()
        // (fixes bug #275)
          MPICommSource_impl( struct gob_cca_common_MPICommSource__object * ior 
            ) : StubBase(ior,true), 
          ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
          ::gob::cca::ports::MPIService((ior==NULL) ? NULL : &((
            *ior).d_gob_cca_ports_mpiservice)),
          ::gob::cca::ports::MPISetup((ior==NULL) ? NULL : &((
            *ior).d_gob_cca_ports_mpisetup)),
          ::gov::cca::Component((ior==NULL) ? NULL : &((
            *ior).d_gov_cca_component)),
        ::gov::cca::ComponentRelease((ior==NULL) ? NULL : &((
          *ior).d_gov_cca_componentrelease)) , _wrapped(false) {_ctor();}


        // user defined construction
        void _ctor();

        // virtual destructor (required)
        virtual ~MPICommSource_impl() { _dtor(); }

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
         *  Get an mpi communicator with the same scope as the component instance.
         * Won't be mpi_comm_null. The communicator returned will be
         * private to the recipient, which implies an mpi_comm_dup by the provider.
         * Call must be made collectively.
         * @return The comm produced, in FORTRAN form. C callers use comm_f2c
         * method defined by their mpi implementation, usually MPI_Comm_f2c,
         * to convert result to MPI_Comm.
         * @throw CCAException if the service cannot be implemented because MPI is
         * not present.
         */
        int64_t
        getComm_impl() // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;

        /**
         *  Let go the communicator. previously fetched with getComm.
         * Call must be made collectively.
         * @throw CCAException if an error is detected.
         */
        void
        releaseComm_impl (
          /* in */int64_t comm
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         *  Get the typically needed basic parallelism information for a component that
         * requires no MPI communication and thus does not need an independent communicator.
         * Rationale: on very large machines, the cost of a Comm_dup should be avoided where possible;
         * The other calls on a MPI Comm object may affect its state, and thus should not
         * be proxied here.
         * @throw CCAException if an error is detected.
         */
        void
        getSizeRank_impl (
          /* out */int64_t& commSize,
          /* out */int64_t& commRank
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         *  Check instance status. Only one init call per instance is allowed.
         * @return true if initAsService or initComponent already done.
         */
        bool
        isInitialized_impl() ;
        /**
         * user defined non-static method.
         */
        void
        initAsInstance_impl (
          /* in */int64_t dupComm,
          /* inout */::gov::cca::AbstractFramework& af
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;


        /**
         *  Set the communicators on an uninitialized mpi support component
         * instance created like any other and register the component through
         * the ServiceRegistry.
         * 
         * In the MPI sense, this call must be collective on the scope of 
         * dupComm.
         * 
         * @param dupComm  the (valid) communicator (in fortran form) to duplicate
         * for those using MPIService.
         * @param af The frame into which the server will add itself.
         * In principle, the caller should be able to forget about the class object
         * they are holding to make this call.
         */
        void
        initAsService_impl (
          /* in */int64_t dupComm
        )
        ;


        /**
         *  Set the communicators on an uninitialized mpi support component
         * instance created like any other.
         * This does NOT cause the component being initialized to register itself
         * as a service for all comers.
         * This method is for treating an instance from inside a frame or
         * subframe as a peer component that may serve only certain other
         * components in the frame, e.g after a comm split.
         * 
         * In the MPI sense, this call must be collective on the scope of
         * dupComm.
         * 
         * @param dupComm  the (valid) communicator (in fortran form) to duplicate
         * for those using MPIService.
         * @param af The frame into which the server will add itself.
         * In principle, the caller should be able to forget about the class object
         * they are holding to make this call.
         */
        void
        initComponent_impl (
          /* in */int64_t dupComm
        )
        ;


        /**
         * Shutdown the previous mpi-related services.
         * @param reclaim if reclaim true, try to release communicator
         * resources allocated in MPIService support.
         * Otherwise, lose them.
         */
        void
        finalize_impl (
          /* in */bool reclaim
        )
        // throws:
        //    ::gov::cca::CCAException
        //    ::sidl::RuntimeException
        ;

      };  // end class MPICommSource_impl

    } // end namespace common
  } // end namespace cca
} // end namespace gob

// DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._hmisc)
// Insert-Code-Here {gob.cca.common.MPICommSource._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._hmisc)

#endif
