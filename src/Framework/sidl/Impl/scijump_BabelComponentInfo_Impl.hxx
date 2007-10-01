// 
// File:          scijump_BabelComponentInfo_Impl.hxx
// Symbol:        scijump.BabelComponentInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.BabelComponentInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_BabelComponentInfo_Impl_hxx
#define included_scijump_BabelComponentInfo_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_BabelComponentInfo_IOR_h
#include "scijump_BabelComponentInfo_IOR.h"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_ComponentInfo_hxx
#include "sci_cca_core_ComponentInfo.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_BabelComponentInfo_hxx
#include "scijump_BabelComponentInfo.hxx"
#endif
#ifndef included_scijump_BabelServices_hxx
#include "scijump_BabelServices.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._hincludes)
//#include <Framework/Core/PortInfoIterator.h>

#include <Core/Thread/Mutex.h>

#include <map>
// DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.BabelComponentInfo" (version 0.2.1)
   */
  class BabelComponentInfo_impl : public virtual ::scijump::BabelComponentInfo 
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._inherits)
  // Insert-Code-Here {scijump.BabelComponentInfo._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._implementation)

    /** The unique name of this component instance. */
    std::string instanceName;

    /** The type of the component. */
    std::string className;

    ::sci::cca::AbstractFramework framework;
    ::gov::cca::TypeMap properties;
    ::gov::cca::Component component;

    scijump::BabelServices services;
    SCIRun::Mutex* lock;

  private:
    bool valid;

//     typedef std::map<std::string, ::sci::cca::core::PortInfo> PortInfoMap;
//     /** ? */
//     class Iterator : public scijump::core::PortInfoIterator {
//       PortInfoMap ports;
//       PortInfoMap::iterator iter;
//     public:
//       Iterator(BabelComponentInfo&);
//       virtual ~Iterator();
//       virtual ::sci::cca::core::PortInfo get();
//       virtual bool done();
//       virtual void next();
//     private:
//       Iterator(const Iterator&);
//       Iterator& operator=(const Iterator&);
//     };
    // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._implementation)

  public:
    // default constructor, used for data wrapping(required)
    BabelComponentInfo_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      BabelComponentInfo_impl( struct scijump_BabelComponentInfo__object * ior 
        ) : StubBase(ior,true), 
      ::gov::cca::ComponentID((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_componentid)),
    ::sci::cca::core::ComponentInfo((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_core_componentinfo)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~BabelComponentInfo_impl() { _dtor(); }

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
      /* in */const ::std::string& instanceName,
      /* in */const ::std::string& className,
      /* in */::sci::cca::AbstractFramework& framework,
      /* in */::gov::cca::Component& component,
      /* in */::scijump::BabelServices& services,
      /* in */::gov::cca::TypeMap& properties
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */const ::std::string& instanceName,
      /* in */const ::std::string& className,
      /* in */::sci::cca::AbstractFramework& framework,
      /* in */::gov::cca::Component& component,
      /* in */::scijump::BabelServices& services,
      /* in */::gov::cca::TypeMap& properties,
      /* in */const ::std::string& serialization
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */const ::std::string& instanceName,
      /* in */const ::std::string& className,
      /* in */::sci::cca::AbstractFramework& framework,
      /* in */::scijump::BabelServices& services,
      /* in */::gov::cca::TypeMap& properties
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    setSerialization_impl (
      /* in */const ::std::string& serialization
    )
    ;

    /**
     * user defined non-static method.
     */
    ::sci::cca::AbstractFramework
    getFramework_impl() ;
    /**
     * user defined non-static method.
     */
    ::gov::cca::Component
    getComponent_impl() ;
    /**
     * user defined non-static method.
     */
    ::gov::cca::Services
    getServices_impl() ;
    /**
     * user defined non-static method.
     */
    ::sidl::array< ::sci::cca::core::PortInfo>
    getPorts_impl() ;
    /**
     * user defined non-static method.
     */
    ::sci::cca::core::PortInfo
    getPortInfo_impl (
      /* in */const ::std::string& portName
    )
    ;

    /**
     * user defined non-static method.
     */
    ::std::string
    getClassName_impl() ;
    /**
     * user defined non-static method.
     */
    ::gov::cca::TypeMap
    getProperties_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setProperties_impl (
      /* in */::gov::cca::TypeMap& properties
    )
    ;

    /**
     * user defined non-static method.
     */
    bool
    callReleaseCallback_impl() ;
    /**
     * user defined non-static method.
     */
    void
    invalidate_impl() ;

    /**
     *  
     * Returns the instance name provided in 
     * <code>BuilderService.createInstance()</code>
     * or in 
     * <code>AbstractFramework.getServices()</code>.
     * @throws CCAException if <code>ComponentID</code> is invalid
     */
    ::std::string
    getInstanceName_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     * Returns a framework specific serialization of the ComponentID.
     * @throws CCAException if <code>ComponentID</code> is
     * invalid.
     */
    ::std::string
    getSerialization_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;
  };  // end class BabelComponentInfo_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._hmisc)
// Insert-Code-Here {scijump.BabelComponentInfo._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._hmisc)

#endif
