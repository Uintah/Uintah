// 
// File:          scijump_core_FrameworkServiceFactory_Impl.hxx
// Symbol:        scijump.core.FrameworkServiceFactory-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.core.FrameworkServiceFactory
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_core_FrameworkServiceFactory_Impl_hxx
#define included_scijump_core_FrameworkServiceFactory_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_core_FrameworkServiceFactory_IOR_h
#include "scijump_core_FrameworkServiceFactory_IOR.h"
#endif
#ifndef included_sci_cca_core_FrameworkServiceFactory_hxx
#include "sci_cca_core_FrameworkServiceFactory.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_core_FrameworkServiceFactory_hxx
#include "scijump_core_FrameworkServiceFactory.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._hincludes)
//#include <Framework/Core/SingletonServiceFactory.h>

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>

#include <Framework/Core/SingletonServiceFactory.h>

#include "scijump.hxx"

#include <string>

// Insert-Code-Here {scijump.core.FrameworkServiceFactory._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._hincludes)

namespace scijump { 
  namespace core { 

    /**
     * Symbol "scijump.core.FrameworkServiceFactory" (version 0.2.1)
     */
    class FrameworkServiceFactory_impl : public virtual 
      ::scijump::core::FrameworkServiceFactory 
    // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._inherits)
    // Insert-Code-Here {scijump.core.FrameworkServiceFactory._inherits} (optional inheritance here)
    // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._inherits)

    {

    // All data marked protected will be accessable by 
    // descendant Impl classes
    protected:

      bool _wrapped;

      // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._implementation)

      scijump::SCIJumpFramework framework; // Babelized framework
      std::string serviceName;
      ServiceFactory* factory;
      // int uses;

//     private:
//       template <typename Service> void initService(const ::std::string& serviceName/*, const ::gov::cca::ComponentID& requester*/);

      // Insert-Code-Here {scijump.core.FrameworkServiceFactory._implementation} (additional details)
      // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._implementation)

    public:
      // default constructor, used for data wrapping(required)
      FrameworkServiceFactory_impl();
      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
        FrameworkServiceFactory_impl( struct 
          scijump_core_FrameworkServiceFactory__object * ior ) : StubBase(ior,
          true), 
      ::sci::cca::core::FrameworkServiceFactory((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_frameworkservicefactory)) , _wrapped(false) {_ctor(
        );}


      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~FrameworkServiceFactory_impl() { _dtor(); }

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
        /* in */void* internalFactoryImpl
      )
      ;

      /**
       * user defined non-static method.
       */
      ::std::string
      getName_impl() ;
      /**
       * user defined non-static method.
       */
      ::sci::cca::core::PortInfo
      getService_impl (
        /* in */const ::std::string& serviceName
      )
      ;

      /**
       * user defined non-static method.
       */
      void
      releaseService_impl (
        /* in */const ::std::string& portName
      )
      ;

    };  // end class FrameworkServiceFactory_impl

  } // end namespace core
} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._hmisc)

// Insert-Code-Here {scijump.core.FrameworkServiceFactory._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._hmisc)

#endif
