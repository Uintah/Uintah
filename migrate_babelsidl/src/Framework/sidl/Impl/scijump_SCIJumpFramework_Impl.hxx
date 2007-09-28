// 
// File:          scijump_SCIJumpFramework_Impl.hxx
// Symbol:        scijump.SCIJumpFramework-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.SCIJumpFramework
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_SCIJumpFramework_Impl_hxx
#define included_scijump_SCIJumpFramework_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_SCIJumpFramework_IOR_h
#include "scijump_SCIJumpFramework_IOR.h"
#endif
#ifndef included_gov_cca_AbstractFramework_hxx
#include "gov_cca_AbstractFramework.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ConnectionID_hxx
#include "gov_cca_ConnectionID.hxx"
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
#ifndef included_sci_cca_core_ServiceInfo_hxx
#include "sci_cca_core_ServiceInfo.hxx"
#endif
#ifndef included_scijump_SCIJumpFramework_hxx
#include "scijump_SCIJumpFramework.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._hincludes)
#include "scijump_core_FrameworkServiceFactory.hxx"
#include <Framework/Core/ComponentModel.h>
#include <Framework/Core/Babel/BabelComponentModel.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>

#include <algorithm>
#include <list>
#include <map>

using namespace SCIRun;

// DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.SCIJumpFramework" (version 0.2.1)
   */
  class SCIJumpFramework_impl : public virtual ::scijump::SCIJumpFramework 
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._inherits)
  // Insert-Code-Here {scijump.SCIJumpFramework._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._implementation)

    typedef std::map<std::string, ::sci::cca::core::FrameworkServiceFactory> FrameworkServiceMap;
    FrameworkServiceMap frameworkServices;

    typedef std::map<std::string, ::gov::cca::Services> ServicesMap;
    ServicesMap services;

    /** The set of registered components available in the framework, indexed by their instance names. */
    //typedef std::map<std::string, ::sci::cca::core::ComponentInfo*> ComponentInstanceMap;
    typedef std::map<std::string, ::sci::cca::core::ComponentInfo> ComponentInstanceMap;
    ComponentInstanceMap components;
    SCIRun::Mutex* lock_components;

    typedef std::list< ::sci::cca::core::ConnectionInfo> ConnectionList;
    ConnectionList connections;
    SCIRun::Mutex* lock_connections;

    void initFrameworkServices();
    bool addFrameworkService(::scijump::core::FrameworkServiceFactory& factory, FrameworkServiceMap& frameworkServices);
    bool removeFrameworkService(const std::string& serviceName, FrameworkServiceMap& frameworkServices);

    //bool addComponent(const std::string& name, ::sci::cca::core::ComponentInfo& ci, ComponentInstanceMap& components);
    //bool removeComponent(const std::string& name, ComponentInstanceMap& components);

    BabelComponentModel* bcm;

  private:
    class ConnectionInfo_eq : public std::unary_function< ::sci::cca::core::ConnectionInfo, bool> {
      ::sci::cca::core::ConnectionInfo ci;
    public:
      explicit ConnectionInfo_eq(::sci::cca::core::ConnectionInfo& c) : ci(c) {}
      bool operator() (::sci::cca::core::ConnectionInfo& c) {
        return ci.getUserPortName() == c.getUserPortName() &&
               ci.getProviderPortName() == c.getProviderPortName() &&
               ci.getUser().getInstanceName() == c.getUser().getInstanceName() &&
               ci.getProvider().getInstanceName() == c.getProvider().getInstanceName();
      }
    };

    // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._implementation)

  public:
    // default constructor, used for data wrapping(required)
    SCIJumpFramework_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      SCIJumpFramework_impl( struct scijump_SCIJumpFramework__object * ior ) : 
        StubBase(ior,true), 
      ::gov::cca::AbstractFramework((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_abstractframework)),
    ::sci::cca::AbstractFramework((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_abstractframework)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~SCIJumpFramework_impl() { _dtor(); }

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
    bool
    isFrameworkService_impl (
      /* in */const ::std::string& name
    )
    ;

    /**
     * user defined non-static method.
     */
    ::scijump::core::ServiceInfo
    getFrameworkService_impl (
      /* in */const ::std::string& serviceName,
      /* in */::sci::cca::core::PortInfo& requesterPort
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    releaseFrameworkService_impl (
      /* in */::sci::cca::core::ServiceInfo& info
    )
    ;


    /**
     *  Eliminates the component instance ``cid'' from the scope of the
     * framework.  The ``timeout'' parameter specifies the maximum allowable
     * wait time for this operation.  A timeout of 0 leaves the wait time up to
     * the framework.  If the destroy operation is not completed in the maximum
     * allowed number of seconds, or the referenced component does not exist,
     * then a CCAException is thrown.
     * 
     * Like createComponentInstance, this method is only intended to be called
     * by the BuilderService class.  It searches the list of registered
     * components (compIDs) for the matching component ID, unregisters it, finds
     * the correct ComponentModel for the type, then calls
     * ComponentModel::destroyInstance to properly destroy the component. 
     */
    void
    destroyComponentInstance_impl (
      /* in */::gov::cca::ComponentID& cid,
      /* in */float timeout
    )
    ;


    /**
     *  Creates an instance of the component defined by the string ``type'',
     * which must uniquely define the type of the component.  The component
     * instance is given the name ``name''.   If the instance name is not
     * specified (i.e. the method is passed an empty string), then the component
     * will be assigned a unique name automatically.
     * 
     * This method is ``semi-private'' and intended to be called only by the
     * BuilderService class.  It works by searching the list of ComponentModels
     * (the ivar \em models) for a matching registered type, and then calling
     * the createInstance method on the appropriate ComponentModel object. 
     */
    ::gov::cca::ComponentID
    createComponentInstance_impl (
      /* in */const ::std::string& name,
      /* in */const ::std::string& className,
      /* in */::gov::cca::TypeMap& tm
    )
    ;

    /**
     * user defined non-static method.
     */
    ::gov::cca::ConnectionID
    createConnectionInstance_impl (
      /* in */::sci::cca::core::ComponentInfo& user,
      /* in */::sci::cca::core::ComponentInfo& provider,
      /* in */const ::std::string& userPortName,
      /* in */const ::std::string& providerPortName,
      /* in */::gov::cca::TypeMap& tm
    )
    ;

    /**
     * user defined non-static method.
     */
    ::gov::cca::ConnectionID
    destroyConnectionInstance_impl (
      /* in */::gov::cca::ConnectionID& connID
    )
    ;


    /**
     * Registers the slave framework with the master framework. Intended to be called
     * only by the representative slave framework process.
     * @param size Total number of parallel slave frameworks.
     * @param slaveURLs Urls of the slave framework.
     * @param slaveName Name of the slave resource.
     * @return A positive number or zero if framework was registered
     * successfully, negative number on error.
     */
    int32_t
    registerLoader_impl (
      /* in */const ::std::string& slaveName,
      /* in array<string> */::sidl::array< ::std::string>& slaveURLs
    )
    ;

    /**
     * user defined non-static method.
     */
    int32_t
    unregisterLoader_impl (
      /* in */const ::std::string& slaveName
    )
    ;


    /**
     *  
     * Create an empty TypeMap. Presumably this would be used in 
     * an ensuing call to <code>getServices()</code>. The "normal" method of
     * creating typemaps is found in the <code>Services</code> interface. It
     * is duplicated here to break the "chicken and egg" problem.
     */
    ::gov::cca::TypeMap
    createTypeMap_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Retrieve a Services handle to the underlying framework. 
     * This interface effectively causes the calling program to 
     * appear as the image of a component inside the framework.
     * This method may be called any number of times
     * with different arguments, creating a new component image 
     * each time. 
     * The only proper method to destroy a Services obtained 
     * from this interface is to pass it to releaseServices.
     * 
     * @param selfInstanceName the Component instance name,
     * as it will appear in the framework.
     * 
     * @param selfClassName the Component type of the 
     * calling program, as it will appear in the framework. 
     * 
     * @param selfProperties (which can be null) the properties 
     * of the component image to appear. 
     * 
     * @throws CCAException in the event that selfInstanceName 
     * is already in use by another component.
     * 
     * @return  A Services object that pertains to the
     * image of the this component. This is identical
     * to the object passed into Component.setServices() 
     * when a component is created.
     */
    ::gov::cca::Services
    getServices_impl (
      /* in */const ::std::string& selfInstanceName,
      /* in */const ::std::string& selfClassName,
      /* in */::gov::cca::TypeMap& selfProperties
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Inform framework that the <code>Services</code> handle is no longer needed by the 
     * caller and that the reference to its component image is to be
     * deleted from the context of the underlying framework. This invalidates
     * any <code>ComponentID</code>'s or <code>ConnectionID</code>'s associated 
     * with the given <code>Services</code>' component image. 
     * 
     * @param services The result of getServices earlier obtained.
     * 
     * @throws CCAException if the <code>Services</code>
     * handle has already been released or is otherwise rendered invalid 
     * or was not obtained from <code>getServices()</code>.
     */
    void
    releaseServices_impl (
      /* in */::gov::cca::Services& services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Tell the framework it is no longer needed and to clean up after itself. 
     * @throws CCAException if the framework has already been shutdown.
     */
    void
    shutdownFramework_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Creates a new framework instance based on the same underlying 
     * framework implementation. This does not copy the existing 
     * framework, nor are any of the user-instantiated components in
     * the original framework available in the newly created 
     * <code>AbstractFramework</code>. 
     * 
     * @throws CCAException when one of the following conditions occur:
     * 
     * (1)the AbstractFramework previously had shutdownFramework() called on it, or 
     * (2)the underlying framework implementation does not permit creation 
     * of another instance.	 
     */
    ::gov::cca::AbstractFramework
    createEmptyFramework_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;
  };  // end class SCIJumpFramework_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._hmisc)
// Insert-Code-Here {scijump.SCIJumpFramework._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._hmisc)

#endif
