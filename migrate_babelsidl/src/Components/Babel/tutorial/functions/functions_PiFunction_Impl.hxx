// 
// File:          functions_PiFunction_Impl.hxx
// Symbol:        functions.PiFunction-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for functions.PiFunction
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_functions_PiFunction_Impl_hxx
#define included_functions_PiFunction_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_functions_PiFunction_IOR_h
#include "functions_PiFunction_IOR.h"
#endif
#ifndef included_function_FunctionPort_hxx
#include "function_FunctionPort.hxx"
#endif
#ifndef included_functions_PiFunction_hxx
#include "functions_PiFunction.hxx"
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


// DO-NOT-DELETE splicer.begin(functions.PiFunction._hincludes)
// Insert-Code-Here {functions.PiFunction._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(functions.PiFunction._hincludes)

namespace functions { 

  /**
   * Symbol "functions.PiFunction" (version 1.0)
   */
  class PiFunction_impl : public virtual ::functions::PiFunction 
  // DO-NOT-DELETE splicer.begin(functions.PiFunction._inherits)
  // Insert-Code-Here {functions.PiFunction._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(functions.PiFunction._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(functions.PiFunction._implementation)
    // Insert-Code-Here {functions.PiFunction._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(functions.PiFunction._implementation)

  public:
    // default constructor, used for data wrapping(required)
    PiFunction_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      PiFunction_impl( struct functions_PiFunction__object * ior ) : StubBase(
        ior,true), 
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
      ::function::FunctionPort((ior==NULL) ? NULL : &((
        *ior).d_function_functionport)),
    ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)) , 
      _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~PiFunction_impl() { _dtor(); }

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
    init_impl (
      /* in array<string> */::sidl::array< ::std::string>& params
    )
    ;

    /**
     * user defined non-static method.
     */
    double
    evaluate_impl (
      /* in */double x
    )
    ;


    /**
     *  Starts up a component presence in the calling framework.
     * @param services the component instance's handle on the framework world.
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
    setServices_impl (
      /* in */::gov::cca::Services& services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class PiFunction_impl

} // end namespace functions

// DO-NOT-DELETE splicer.begin(functions.PiFunction._hmisc)
// Insert-Code-Here {functions.PiFunction._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(functions.PiFunction._hmisc)

#endif
