// 
// File:          scijump_BabelConnectionInfo_Impl.hxx
// Symbol:        scijump.BabelConnectionInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.BabelConnectionInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_BabelConnectionInfo_Impl_hxx
#define included_scijump_BabelConnectionInfo_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_BabelConnectionInfo_IOR_h
#include "scijump_BabelConnectionInfo_IOR.h"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_core_ComponentInfo_hxx
#include "sci_cca_core_ComponentInfo.hxx"
#endif
#ifndef included_sci_cca_core_ConnectionInfo_hxx
#include "sci_cca_core_ConnectionInfo.hxx"
#endif
#ifndef included_scijump_BabelConnectionInfo_hxx
#include "scijump_BabelConnectionInfo.hxx"
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


// DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._hincludes)
#include <Core/Thread/Mutex.h>
// DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.BabelConnectionInfo" (version 0.2.1)
   */
  class BabelConnectionInfo_impl : public virtual 
    ::scijump::BabelConnectionInfo 
  // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._inherits)
  // Insert-Code-Here {scijump.BabelConnectionInfo._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._implementation)
    ::sci::cca::core::ComponentInfo user;
    ::sci::cca::core::ComponentInfo provider;
    ::std::string userPortName;
    ::std::string providerPortName;
    ::gov::cca::TypeMap properties;
    SCIRun::Mutex* lock;

  private:
    bool valid;
    // DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._implementation)

  public:
    // default constructor, used for data wrapping(required)
    BabelConnectionInfo_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      BabelConnectionInfo_impl( struct scijump_BabelConnectionInfo__object * 
        ior ) : StubBase(ior,true), 
      ::gov::cca::ConnectionID((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_connectionid)),
    ::sci::cca::core::ConnectionInfo((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_core_connectioninfo)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~BabelConnectionInfo_impl() { _dtor(); }

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
      /* in */::sci::cca::core::ComponentInfo& user,
      /* in */::sci::cca::core::ComponentInfo& provider,
      /* in */const ::std::string& userPortName,
      /* in */const ::std::string& providerPortName,
      /* in */::gov::cca::TypeMap& properties
    )
    ;

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
    void
    invalidate_impl() ;

    /**
     *  
     * Get the providing component (callee) ID.
     * @return ComponentID of the component that has 
     * provided the Port for this connection. 
     * @throws CCAException if the underlying connection 
     * is no longer valid.
     */
    ::gov::cca::ComponentID
    getProvider_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Get the using component (caller) ID.
     * @return ComponentID of the component that is using the provided Port.
     * @throws CCAException if the underlying connection is no longer valid.
     */
    ::gov::cca::ComponentID
    getUser_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Get the port name in the providing component of this connection.
     * @return the instance name of the provided Port.
     * @throws CCAException if the underlying connection is no longer valid.
     */
    ::std::string
    getProviderPortName_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     *  
     * Get the port name in the using component of this connection.
     * Return the instance name of the Port registered for use in 
     * this connection.
     * @throws CCAException if the underlying connection is no longer valid.
     */
    ::std::string
    getUserPortName_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;
  };  // end class BabelConnectionInfo_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.BabelConnectionInfo._hmisc)
// Insert-Code-Here {scijump.BabelConnectionInfo._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.BabelConnectionInfo._hmisc)

#endif
