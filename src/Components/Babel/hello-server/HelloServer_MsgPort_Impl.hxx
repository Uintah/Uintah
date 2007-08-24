// 
// File:          HelloServer_MsgPort_Impl.hxx
// Symbol:        HelloServer.MsgPort-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for HelloServer.MsgPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_HelloServer_MsgPort_Impl_hxx
#define included_HelloServer_MsgPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_HelloServer_MsgPort_IOR_h
#include "HelloServer_MsgPort_IOR.h"
#endif
#ifndef included_HelloServer_MsgPort_hxx
#include "HelloServer_MsgPort.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
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


// DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._hincludes)
// Insert-Code-Here {HelloServer.MsgPort._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(HelloServer.MsgPort._hincludes)

namespace HelloServer { 

  /**
   * Symbol "HelloServer.MsgPort" (version 1.0)
   */
  class MsgPort_impl : public virtual ::HelloServer::MsgPort 
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._inherits)
  // Insert-Code-Here {HelloServer.MsgPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._implementation)
    // Insert-Code-Here {HelloServer.MsgPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._implementation)

  public:
    // default constructor, used for data wrapping(required)
    MsgPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      MsgPort_impl( struct HelloServer_MsgPort__object * ior ) : StubBase(ior,
        true), 
    ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)) , _wrapped(
      false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~MsgPort_impl() { _dtor(); }

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
    printMsg_impl (
      /* in */const ::std::string& msg
    )
    ;

  };  // end class MsgPort_impl

} // end namespace HelloServer

// DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._hmisc)
// Insert-Code-Here {HelloServer.MsgPort._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(HelloServer.MsgPort._hmisc)

#endif
