// 
// File:          who_IDPort_Impl.hxx
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_who_IDPort_Impl_hxx
#define included_who_IDPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_who_IDPort_IOR_h
#include "who_IDPort_IOR.h"
#endif
#ifndef included_gov_cca_ports_IDPort_hxx
#include "gov_cca_ports_IDPort.hxx"
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
#ifndef included_who_IDPort_hxx
#include "who_IDPort.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.IDPort._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.IDPort._hincludes)

namespace who { 

  /**
   * Symbol "who.IDPort" (version 1.0)
   */
  class IDPort_impl : public virtual ::who::IDPort 
  // DO-NOT-DELETE splicer.begin(who.IDPort._inherits)
  // Insert-Code-Here {who.IDPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.IDPort._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(who.IDPort._implementation)
    // Insert-Code-Here {who.IDPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(who.IDPort._implementation)

  public:
    // default constructor, used for data wrapping(required)
    IDPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      IDPort_impl( struct who_IDPort__object * ior ) : StubBase(ior,true), 
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::gov::cca::ports::IDPort((ior==NULL) ? NULL : &((
      *ior).d_gov_cca_ports_idport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~IDPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     *  Test prot. Return a string as an ID for Hello component
     */
    ::std::string
    getID_impl() ;
  };  // end class IDPort_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.IDPort._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(who.IDPort._hmisc)

#endif
