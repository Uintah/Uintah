// 
// File:          scijump_core_PortInfo_Impl.hxx
// Symbol:        scijump.core.PortInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.core.PortInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_core_PortInfo_Impl_hxx
#define included_scijump_core_PortInfo_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_core_PortInfo_IOR_h
#include "scijump_core_PortInfo_IOR.h"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_core_NotInitializedException_hxx
#include "sci_cca_core_NotInitializedException.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_sci_cca_core_PortType_hxx
#include "sci_cca_core_PortType.hxx"
#endif
#ifndef included_scijump_core_PortInfo_hxx
#include "scijump_core_PortInfo.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._hincludes)

#include <vector>

// Insert-Code-Here {scijump.core.PortInfo._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.core.PortInfo._hincludes)

namespace scijump { 
  namespace core { 

    /**
     * Symbol "scijump.core.PortInfo" (version 0.2.1)
     */
    class PortInfo_impl : public virtual ::scijump::core::PortInfo 
    // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._inherits)
    // Insert-Code-Here {scijump.core.PortInfo._inherits} (optional inheritance here)
    // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._inherits)

    {

    // All data marked protected will be accessable by 
    // descendant Impl classes
    protected:

      bool _wrapped;

      // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._implementation)

      ::gov::cca::Port port;
      ::sci::cca::core::PortType portType;
      std::string name;
      std::string className;

    private:
      std::vector< ::sci::cca::core::PortInfo> connections;

      ::gov::cca::TypeMap properties;
      int useCount;
      //SCIRun::Mutex lock;

      // Insert-Code-Here {scijump.core.PortInfo._implementation} (additional details)
      // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._implementation)

    public:
      // default constructor, used for data wrapping(required)
      PortInfo_impl();
      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
        PortInfo_impl( struct scijump_core_PortInfo__object * ior ) : StubBase(
          ior,true), 
      ::sci::cca::core::PortInfo((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_portinfo)) , _wrapped(false) {_ctor();}


      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~PortInfo_impl() { _dtor(); }

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
        /* in */::gov::cca::Port& port,
        /* in */const ::std::string& name,
        /* in */const ::std::string& className,
        /* in */::sci::cca::core::PortType portType,
        /* in */::gov::cca::TypeMap& properties
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      initialize_impl (
        /* in */const ::std::string& name,
        /* in */const ::std::string& className,
        /* in */::sci::cca::core::PortType portType,
        /* in */::gov::cca::TypeMap& properties
      )
      ;

      /**
       * user defined non-static method.
       */
      bool
      connect_impl (
        /* in */::sci::cca::core::PortInfo& to
      )
      ;

      /**
       * user defined non-static method.
       */
      bool
      disconnect_impl (
        /* in */::sci::cca::core::PortInfo& peer
      )
      ;

      /**
       * user defined non-static method.
       */
      bool
      available_impl() ;
      /**
       * user defined non-static method.
       */
      bool
      canConnectTo_impl (
        /* in */::sci::cca::core::PortInfo& toPortInfo
      )
      ;

      /**
       * user defined non-static method.
       */
      bool
      isConnected_impl() ;
      /**
       * user defined non-static method.
       */
      bool
      inUse_impl() ;
      /**
       * user defined non-static method.
       */
      int32_t
      numOfConnections_impl() ;
      /**
       * user defined non-static method.
       */
      ::gov::cca::TypeMap
      getProperties_impl() ;
      /**
       * user defined non-static method.
       */
      ::gov::cca::Port
      getPort_impl() // throws:
      //     ::sci::cca::core::NotInitializedException
      //     ::sidl::RuntimeException
      ;
      /**
       * user defined non-static method.
       */
      ::sci::cca::core::PortInfo
      getPeer_impl() // throws:
      //     ::gov::cca::CCAException
      //     ::sidl::RuntimeException
      ;
      /**
       * user defined non-static method.
       */
      ::sci::cca::core::PortType
      getPortType_impl() ;
      /**
       * user defined non-static method.
       */
      ::std::string
      getName_impl() ;
      /**
       * user defined non-static method.
       */
      ::std::string
      getClass_impl() ;
      /**
       * user defined non-static method.
       */
      void
      incrementUseCount_impl() ;
      /**
       * user defined non-static method.
       */
      bool
      decrementUseCount_impl() ;
    };  // end class PortInfo_impl

  } // end namespace core
} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._hmisc)
// Insert-Code-Here {scijump.core.PortInfo._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.core.PortInfo._hmisc)

#endif
