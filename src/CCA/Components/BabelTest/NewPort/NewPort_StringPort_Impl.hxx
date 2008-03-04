// 
// File:          NewPort_StringPort_Impl.hxx
// Symbol:        NewPort.StringPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for NewPort.StringPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_NewPort_StringPort_Impl_hxx
#define included_NewPort_StringPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_NewPort_StringPort_IOR_h
#include "NewPort_StringPort_IOR.h"
#endif
#ifndef included_NewPort_StringPort_hxx
#include "NewPort_StringPort.hxx"
#endif
#ifndef included_NewPort_iStringPort_hxx
#include "NewPort_iStringPort.hxx"
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


// DO-NOT-DELETE splicer.begin(NewPort.StringPort._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._hincludes)

namespace NewPort { 

  /**
   * Symbol "NewPort.StringPort" (version 1.0)
   */
  class StringPort_impl : public virtual ::NewPort::StringPort 
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._inherits)
  // Insert-Code-Here {NewPort.StringPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(NewPort.StringPort._implementation)
    // Insert-Code-Here {NewPort.StringPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(NewPort.StringPort._implementation)

  public:
    // default constructor, used for data wrapping(required)
    StringPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      StringPort_impl( struct NewPort_StringPort__object * ior ) : StubBase(ior,
        true), 
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::NewPort::iStringPort((ior==NULL) ? NULL : &((
      *ior).d_newport_istringport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~StringPort_impl() { _dtor(); }

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
    ::std::string
    getString_impl() ;
  };  // end class StringPort_impl

} // end namespace NewPort

// DO-NOT-DELETE splicer.begin(NewPort.StringPort._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._hmisc)

#endif
