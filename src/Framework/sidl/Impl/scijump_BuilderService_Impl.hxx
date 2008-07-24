// 
// File:          scijump_BuilderService_Impl.hxx
// Symbol:        scijump.BuilderService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.BuilderService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_BuilderService_Impl_hxx
#define included_scijump_BuilderService_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_BuilderService_IOR_h
#include "scijump_BuilderService_IOR.h"
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
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_Event_hxx
#include "sci_cca_Event.hxx"
#endif
#ifndef included_sci_cca_EventListener_hxx
#include "sci_cca_EventListener.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sci_cca_ports_BuilderService_hxx
#include "sci_cca_ports_BuilderService.hxx"
#endif
#ifndef included_scijump_BuilderService_hxx
#include "scijump_BuilderService.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.BuilderService._hincludes)

#include "scijump.hxx"

// Insert-Code-Here {scijump.BuilderService._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(scijump.BuilderService._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.BuilderService" (version 0.2.1)
   */
  class BuilderService_impl : public virtual ::scijump::BuilderService 
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService._inherits)
  // Insert-Code-Here {scijump.BuilderService._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.BuilderService._implementation)

    scijump::SCIJumpFramework framework;

    // DO-NOT-DELETE splicer.end(scijump.BuilderService._implementation)

  public:
    // default constructor, used for data wrapping(required)
    BuilderService_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      BuilderService_impl( struct scijump_BuilderService__object * ior ) : 
        StubBase(ior,true), 
      ::sci::cca::EventListener((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_eventlistener)),
      ::sci::cca::core::FrameworkService((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_frameworkservice)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
      ::gov::cca::ports::BuilderService((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_ports_builderservice)),
    ::sci::cca::ports::BuilderService((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_ports_builderservice)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~BuilderService_impl() { _dtor(); }

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
     *  Get available ports in c2 that can be connected to port1 of c1. 
     */
    ::sidl::array< ::std::string>
    getCompatiblePortList_impl (
      /* in */::gov::cca::ComponentID& c1,
      /* in */const ::std::string& port1,
      /* in */::gov::cca::ComponentID& c2
    )
    ;


    /**
     * TODO: document getBridgeablePortList
     * @param cid1
     * @param port1
     * @param  cid2
     */
    ::sidl::array< ::std::string>
    getBridgeablePortList_impl (
      /* in */::gov::cca::ComponentID& cid1,
      /* in */const ::std::string& port1,
      /* in */::gov::cca::ComponentID& cid2
    )
    ;

    /**
     * user defined non-static method.
     */
    ::std::string
    generateBridge_impl (
      /* in */::gov::cca::ComponentID& user,
      /* in */const ::std::string& usingPortName,
      /* in */::gov::cca::ComponentID& provider,
      /* in */const ::std::string& providingPortName
    )
    ;


    /**
     * Creates an instance of a CCA component of the type defined by the 
     * string className.  The string classname uniquely defines the
     * "type" of the component, e.g.
     * doe.cca.Library.GaussianElmination. 
     * It has an instance name given by the string instanceName.
     * The instanceName may be empty (zero length) in which case
     * the instanceName will be assigned to the component automatically.
     * @throws CCAException If the Component className is unknown, or if the
     * instanceName has already been used, a CCAException is thrown.
     * @return A ComponentID corresponding to the created component. Destroying
     * the returned ID does not destroy the component; 
     * see destroyInstance instead.
     */
    ::gov::cca::ComponentID
    createInstance_impl (
      /* in */const ::std::string& instanceName,
      /* in */const ::std::string& className,
      /* in */::gov::cca::TypeMap& properties
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Get component list.
     * @return a ComponentID for each component currently created.
     */
    ::sidl::array< ::gov::cca::ComponentID>
    getComponentIDs_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Get property map for component.
     * @return the public properties associated with the component referred to by
     * ComponentID. 
     * @throws a CCAException if the ComponentID is invalid.
     */
    ::gov::cca::TypeMap
    getComponentProperties_impl (
      /* in */::gov::cca::ComponentID& cid
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     * Causes the framework implementation to associate the given properties 
     * with the component designated by cid. 
     * @throws CCAException if cid is invalid or if there is an attempted
     * change to a property locked by the framework implementation.
     */
    void
    setComponentProperties_impl (
      /* in */::gov::cca::ComponentID& cid,
      /* in */::gov::cca::TypeMap& map
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Get component id from stringified reference.
     * @return a ComponentID from the string produced by 
     * ComponentID.getSerialization(). 
     * @throws CCAException if the string does not represent the appropriate 
     * serialization of a ComponentID for the underlying framework.
     */
    ::gov::cca::ComponentID
    getDeserialization_impl (
      /* in */const ::std::string& s
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Get id from name by which it was created.
     * @return a ComponentID from the instance name of the component
     * produced by ComponentID.getInstanceName().
     * @throws CCAException if there is no component matching the 
     * given componentInstanceName.
     */
    ::gov::cca::ComponentID
    getComponentID_impl (
      /* in */const ::std::string& componentInstanceName
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Eliminate the Component instance, from the scope of the framework.
     * @param toDie the component to be removed.
     * @param timeout the allowable wait; 0 means up to the framework.
     * @throws CCAException if toDie refers to an invalid component, or
     * if the operation takes longer than timeout seconds.
     */
    void
    destroyInstance_impl (
      /* in */::gov::cca::ComponentID& toDie,
      /* in */float timeout
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Get the names of Port instances provided by the identified component.
     * @param cid the component.
     * @throws CCAException if cid refers to an invalid component.
     */
    ::sidl::array< ::std::string>
    getProvidedPortNames_impl (
      /* in */::gov::cca::ComponentID& cid
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Get the names of Port instances used by the identified component.
     * @param cid the component.
     * @throws CCAException if cid refers to an invalid component. 
     */
    ::sidl::array< ::std::string>
    getUsedPortNames_impl (
      /* in */::gov::cca::ComponentID& cid
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Fetch map of Port properties exposed by the framework.
     * @return the public properties pertaining to the Port instance 
     * portname on the component referred to by cid. 
     * @throws CCAException when any one of the following conditions occur:<ul>
     * <li>portname is not a registered Port on the component indicated by cid,
     * <li>cid refers to an invalid component. </ul>
     */
    ::gov::cca::TypeMap
    getPortProperties_impl (
      /* in */::gov::cca::ComponentID& cid,
      /* in */const ::std::string& portName
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  
     * Associates the properties given in map with the Port indicated by 
     * portname. The component must have a Port known by portname.
     * @throws CCAException if either cid or portname are
     * invalid, or if this a changed property is locked by 
     * the underlying framework or component.
     */
    void
    setPortProperties_impl (
      /* in */::gov::cca::ComponentID& cid,
      /* in */const ::std::string& portName,
      /* in */::gov::cca::TypeMap& map
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     * Creates a connection between ports on component user and 
     * component provider. Destroying the ConnectionID does not
     * cause a disconnection; for that, see disconnect().
     * @throws CCAException when any one of the following conditions occur:<ul>
     * <li>If either user or provider refer to an invalid component,
     * <li>If either usingPortName or providingPortName refer to a 
     * nonexistent Port on their respective component,
     * <li>If other-- In reality there are a lot of things that can go wrong 
     * with this operation, especially if the underlying connections 
     * involve networking.</ul>
     */
    ::gov::cca::ConnectionID
    connect_impl (
      /* in */::gov::cca::ComponentID& user,
      /* in */const ::std::string& usingPortName,
      /* in */::gov::cca::ComponentID& provider,
      /* in */const ::std::string& providingPortName
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Returns a list of connections as an array of 
     * handles. This will return all connections involving components 
     * in the given componentList of ComponentIDs. This
     * means that ConnectionID's will be returned even if only one 
     * of the participants in the connection appears in componentList.
     * 
     * @throws CCAException if any component in componentList is invalid.
     */
    ::sidl::array< ::gov::cca::ConnectionID>
    getConnectionIDs_impl (
      /* in array<gov.cca.ComponentID> */::sidl::array< 
        ::gov::cca::ComponentID>& componentList
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     * Fetch property map of a connection.
     * @returns the properties for the given connection.
     * @throws CCAException if connID is invalid.
     */
    ::gov::cca::TypeMap
    getConnectionProperties_impl (
      /* in */::gov::cca::ConnectionID& connID
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Associates the properties with the connection.
     * @param map the source of the properties.
     * @param connID connection to receive property values.
     * @throws CCAException if connID is invalid, or if this changes 
     * a property locked by the underlying framework.
     */
    void
    setConnectionProperties_impl (
      /* in */::gov::cca::ConnectionID& connID,
      /* in */::gov::cca::TypeMap& map
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Disconnect the connection indicated by connID before the indicated
     * timeout in secs. Upon successful completion, connID and the connection
     * it represents become invalid. 
     * @param timeout the time in seconds to wait for a connection to close; 0
     * means to use the framework implementation default.
     * @param connID the connection to be broken.
     * @throws CCAException when any one of the following conditions occur: <ul>
     * <li>id refers to an invalid ConnectionID,
     * <li>timeout is exceeded, after which, if id was valid before 
     * disconnect() was invoked, it remains valid
     * </ul>
     */
    void
    disconnect_impl (
      /* in */::gov::cca::ConnectionID& connID,
      /* in */float timeout
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  Remove all connections between components id1 and id2 within 
     * the period of timeout secs. If id2 is null, then all connections 
     * to id1 are removed (within the period of timeout secs).
     * @throws CCAException when any one of the following conditions occur:<ul>
     * <li>id1 or id2 refer to an invalid ComponentID (other than id2 == null),
     * <li>The timeout period is exceeded before the disconnections can be made. 
     * </ul>
     */
    void
    disconnectAll_impl (
      /* in */::gov::cca::ComponentID& id1,
      /* in */::gov::cca::ComponentID& id2,
      /* in */float timeout
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;


    /**
     *  This is where event processing by a listener takes place. This
     * is a call-back method that a topic subscriber implements and
     * gets called for each new event.
     * 
     * @topicName - The topic for which the Event was created and sent.
     * @theEvent - The payload.
     */
    void
    processEvent_impl (
      /* in */const ::std::string& topicName,
      /* in */::sci::cca::Event& theEvent
    )
    ;

  };  // end class BuilderService_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.BuilderService._hmisc)
// Insert-Code-Here {scijump.BuilderService._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.BuilderService._hmisc)

#endif
