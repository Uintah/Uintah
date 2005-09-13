// 
// File:          NewPort_Com_Impl.hh
// Symbol:        NewPort.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for NewPort.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_NewPort_Com_Impl_hh
#define included_NewPort_Com_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_NewPort_Com_IOR_h
#include "NewPort_Com_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_NewPort_Com_hh
#include "NewPort_Com.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(NewPort.Com._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(NewPort.Com._includes)

namespace NewPort { 

  /**
   * Symbol "NewPort.Com" (version 1.0)
   */
  class Com_impl
  // DO-NOT-DELETE splicer.begin(NewPort.Com._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(NewPort.Com._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Com self;

    // DO-NOT-DELETE splicer.begin(NewPort.Com._implementation)
    gov::cca::Services svc;
    // DO-NOT-DELETE splicer.end(NewPort.Com._implementation)

  private:
    // private default constructor (required)
    Com_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Com_impl( struct NewPort_Com__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Com_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:


    /**
     * Starts up a component presence in the calling framework.
     * @param Svc the component instance's handle on the framework world.
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
    setServices (
      /* in */ ::gov::cca::Services services
    )
    throw () 
    ;

  };  // end class Com_impl

} // end namespace NewPort

// DO-NOT-DELETE splicer.begin(NewPort.Com._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(NewPort.Com._misc)

#endif
