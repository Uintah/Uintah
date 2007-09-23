// 
// File:          randomgens_RandNumGenerator_Impl.cxx
// Symbol:        randomgens.RandNumGenerator-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for randomgens.RandNumGenerator
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "randomgens_RandNumGenerator_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._includes)
// Insert-Code-Here {randomgens.RandNumGenerator._includes} (additional includes or code)

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

// DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
randomgens::RandNumGenerator_impl::RandNumGenerator_impl() : StubBase(
  reinterpret_cast< void*>(::randomgens::RandNumGenerator::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._ctor2)
  // Insert-Code-Here {randomgens.RandNumGenerator._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._ctor2)
}

// user defined constructor
void randomgens::RandNumGenerator_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._ctor)
  // Insert-Code-Here {randomgens.RandNumGenerator._ctor} (constructor)

   // Set initial seed 
   srand( (unsigned)time( NULL ) );

  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._ctor)
}

// user defined destructor
void randomgens::RandNumGenerator_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._dtor)
  // Insert-Code-Here {randomgens.RandNumGenerator._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._dtor)
}

// static class initializer
void randomgens::RandNumGenerator_impl::_load() {
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._load)
  // Insert-Code-Here {randomgens.RandNumGenerator._load} (class initialization)
  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getRandomNumber[]
 */
double
randomgens::RandNumGenerator_impl::getRandomNumber_impl () 

{
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator.getRandomNumber)
  // Insert-Code-Here {randomgens.RandNumGenerator.getRandomNumber} (getRandomNumber method)

  double random_value = static_cast < double >(rand ());
  return random_value / RAND_MAX;

  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator.getRandomNumber)
}

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
randomgens::RandNumGenerator_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator.setServices)
  // Insert-Code-Here {randomgens.RandNumGenerator.setServices} (setServices method)

  frameworkServices = services;
  
  if(frameworkServices._not_nil()) {
    gov::cca::TypeMap tm = frameworkServices.createTypeMap();
    if(tm._is_nil()) {
      fprintf(stderr, "%s:%d: gov::cca::TypeMap is nil\n",
          __FILE__, __LINE__);
    }
    
    gov::cca::Port p = (*this); // Babel-required cast
    
    if(p._is_nil()) {
      fprintf(stderr, "%s:%d: p is nil\n", __FILE__, __LINE__);
    } 
    
    frameworkServices.addProvidesPort(p,
                                      "RandomGeneratorPort-pp",
                                      "randomgen.RandomGeneratorPort",
                                      tm);
  }
  
  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator.setServices)
}


// DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._misc)
// Insert-Code-Here {randomgens.RandNumGenerator._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._misc)

