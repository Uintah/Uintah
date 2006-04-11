// 
// File:          framework_Component_Impl.cxx
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "framework_Component_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(framework.Component._includes)
// Insert-Code-Here {framework.Component._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(framework.Component._includes)

// user defined constructor
void framework::Component_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._ctor)
  // Insert-Code-Here {framework.Component._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(framework.Component._ctor)
}

// user defined destructor
void framework::Component_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._dtor)
  // Insert-Code-Here {framework.Component._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(framework.Component._dtor)
}

// static class initializer
void framework::Component_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.Component._load)
  // Insert-Code-Here {framework.Component._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.Component._load)
}

// user defined static methods: (none)

// user defined non-static methods:
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
framework::Component_impl::setServices_impl (
  /* in */UCXX ::gov::cca::Services services ) 
{
  // DO-NOT-DELETE splicer.begin(framework.Component.setServices)
  // Insert-Code-Here {framework.Component.setServices} (setServices method)
  // DO-NOT-DELETE splicer.end(framework.Component.setServices)
}


// DO-NOT-DELETE splicer.begin(framework.Component._misc)
// Insert-Code-Here {framework.Component._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(framework.Component._misc)

