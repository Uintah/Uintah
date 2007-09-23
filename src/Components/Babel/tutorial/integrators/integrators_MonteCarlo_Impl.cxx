// 
// File:          integrators_MonteCarlo_Impl.cxx
// Symbol:        integrators.MonteCarlo-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for integrators.MonteCarlo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "integrators_MonteCarlo_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._includes)
#include<iostream>
//#include "../functions/function_FunctionPort.hxx"
// DO-NOT-DELETE splicer.end(integrators.MonteCarlo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
integrators::MonteCarlo_impl::MonteCarlo_impl() : StubBase(reinterpret_cast< 
  void*>(::integrators::MonteCarlo::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._ctor2)
  // Insert-Code-Here {integrators.MonteCarlo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._ctor2)
}

// user defined constructor
void integrators::MonteCarlo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._ctor)
  // Insert-Code-Here {integrators.MonteCarlo._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._ctor)
}

// user defined destructor
void integrators::MonteCarlo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._dtor)
  // Insert-Code-Here {integrators.MonteCarlo._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._dtor)
}

// static class initializer
void integrators::MonteCarlo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._load)
  // Insert-Code-Here {integrators.MonteCarlo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  integrate[]
 */
double
integrators::MonteCarlo_impl::integrate_impl (
  /* in */double lowBound,
  /* in */double upBound,
  /* in */int32_t count ) 
{
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo.integrate)
  
  gov::cca::Port fport = frameworkServices.getPort("FunctionPort-up");
  ::function::FunctionPort function = babel_cast< ::function::FunctionPort >(fport);

  gov::cca::Port rport = frameworkServices.getPort("RandomGeneratorPort-up");
  ::randomgen::RandomGeneratorPort rand = babel_cast< ::randomgen::RandomGeneratorPort >(rport);

  if(function._is_nil()||rand._is_nil()) {
    fprintf(stdout, "integrator.MonteCarlo not connected\n");
    frameworkServices.releasePort("FunctionPort-up");
    frameworkServices.releasePort("RandomGeneratorPort-up");
    return -1;    
  }

  double r = 0.0;
  double sum,width,x,func;
  sum=width=x=func=0.0;
  width = upBound - lowBound;
  for(int i=0;i<count;i++){
    x = rand.getRandomNumber();
    x = lowBound + width * x;
    func = function.evaluate(x);
    sum += func;
  }
  r = width*sum/count;
  return r;


  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo.integrate)
}

/**
 * Method:  setServices[]
 */
void
integrators::MonteCarlo_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo.setServices)
  // Insert-Code-Here {integrators.MonteCarlo.setServices} (setServices method)
  
  frameworkServices = services;
  
  // Provide a Function port
  if(services._not_nil()) {
    gov::cca::TypeMap tm = services.createTypeMap();
    if(tm._is_nil()) {
      fprintf(stderr, "%s:%d: gov::cca::TypeMap is nil\n",
          __FILE__, __LINE__);
    } 
    
    gov::cca::Port p = (*this);      //  Babel required casting
    
    if(p._is_nil()) {
      fprintf(stderr, "p is nil");
    } 
    
    services.addProvidesPort(p,"IntegratorPort-pp",
                             "integrator.IntegratorPort", tm);
    services.registerUsesPort ("FunctionPort-up", 
				            "function.FunctionPort", 
				            services.createTypeMap());
  
    services.registerUsesPort ("RandomGeneratorPort-up", 
				            "randomgen.RandomGeneratorPort", 
				            services.createTypeMap());
  
  }


  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo.setServices)
}

/**
 * Method:  releaseServices[]
 */
void
integrators::MonteCarlo_impl::releaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo.releaseServices)
  // Insert-Code-Here {integrators.MonteCarlo.releaseServices} (releaseServices method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(integrators.MonteCarlo.releaseServices)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "releaseServices");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(integrators.MonteCarlo.releaseServices)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo.releaseServices)
}


// DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._misc)
// Insert-Code-Here {integrators.MonteCarlo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(integrators.MonteCarlo._misc)

