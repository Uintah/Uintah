// 
// File:          drivers_CXXDriver_Impl.cxx
// Symbol:        drivers.CXXDriver-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for drivers.CXXDriver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "drivers_CXXDriver_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(drivers.CXXDriver._includes)
// Insert-Code-Here {drivers.CXXDriver._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(drivers.CXXDriver._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
drivers::CXXDriver_impl::CXXDriver_impl() : StubBase(reinterpret_cast< void*>(
  ::drivers::CXXDriver::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._ctor2)
  // Insert-Code-Here {drivers.CXXDriver._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(drivers.CXXDriver._ctor2)
}

// user defined constructor
void drivers::CXXDriver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._ctor)
  // Insert-Code-Here {drivers.CXXDriver._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(drivers.CXXDriver._ctor)
}

// user defined destructor
void drivers::CXXDriver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._dtor)
  // Insert-Code-Here {drivers.CXXDriver._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(drivers.CXXDriver._dtor)
}

// static class initializer
void drivers::CXXDriver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._load)
  // Insert-Code-Here {drivers.CXXDriver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(drivers.CXXDriver._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  go[]
 */
int32_t
drivers::CXXDriver_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver.go)
  // Insert-Code-Here {drivers.CXXDriver.go} (go method)

  double value;
  int count = 100000;
  double lowerBound = 0.0, upperBound = 1.0;

  ::integrator::IntegratorPort integrator;

  // get the port ...
  gov::cca::Port port = frameworkServices.getPort("IntegratorPort-up");
  integrator = babel_cast< ::integrator::IntegratorPort >(port);

  if(integrator._is_nil()) {
    fprintf(stdout, "drivers.CXXDriver not connected\n");
    frameworkServices.releasePort("IntegratorPort-up");
    return -1;    
  }
    // operate on the port
    value = integrator.integrate (lowerBound, upperBound, count);

    fprintf(stdout,"Value = %lf\n", value);
    fflush(stdout);
  
  // release the port.
  frameworkServices.releasePort("IntegratorPort-up");
  return 0;

  // DO-NOT-DELETE splicer.end(drivers.CXXDriver.go)
}

/**
 * Method:  setServices[]
 */
void
drivers::CXXDriver_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver.setServices)
  // Insert-Code-Here {drivers.CXXDriver.setServices} (setServices method)

  frameworkServices = services;
  
  // Provide a Go port
   gov::cca::Port gp = (*this);
        
  frameworkServices.addProvidesPort(gp, 
				        "GoPort", 
				        "gov.cca.ports.GoPort",
				        frameworkServices.createTypeMap());
              
  // Use an IntegratorPort port
  frameworkServices.registerUsesPort ("IntegratorPort-up", 
				            "integrator.IntegratorPort", 
				            frameworkServices.createTypeMap());

  // DO-NOT-DELETE splicer.end(drivers.CXXDriver.setServices)
}


// DO-NOT-DELETE splicer.begin(drivers.CXXDriver._misc)
// Insert-Code-Here {drivers.CXXDriver._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(drivers.CXXDriver._misc)

