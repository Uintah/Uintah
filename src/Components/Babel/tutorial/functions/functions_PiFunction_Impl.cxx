// 
// File:          functions_PiFunction_Impl.cxx
// Symbol:        functions.PiFunction-v1.0
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for functions.PiFunction
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "functions_PiFunction_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(functions.PiFunction._includes)
// Insert-Code-Here {functions.PiFunction._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(functions.PiFunction._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
functions::PiFunction_impl::PiFunction_impl() : StubBase(reinterpret_cast< 
  void*>(::functions::PiFunction::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(functions.PiFunction._ctor2)
  // Insert-Code-Here {functions.PiFunction._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(functions.PiFunction._ctor2)
}

// user defined constructor
void functions::PiFunction_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(functions.PiFunction._ctor)
  // Insert-Code-Here {functions.PiFunction._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(functions.PiFunction._ctor)
}

// user defined destructor
void functions::PiFunction_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(functions.PiFunction._dtor)
  // Insert-Code-Here {functions.PiFunction._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(functions.PiFunction._dtor)
}

// static class initializer
void functions::PiFunction_impl::_load() {
  // DO-NOT-DELETE splicer.begin(functions.PiFunction._load)
  // Insert-Code-Here {functions.PiFunction._load} (class initialization)
  // DO-NOT-DELETE splicer.end(functions.PiFunction._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  init_pre[]
 */
void
functions::PiFunction_impl::init_pre_impl (
  /* in array<string> */::sidl::array< ::std::string>& params ) 
{
  gov::cca::TypeMap evtMap = typemap.cloneEmpty();
  evtMap.putStringArray("params",params);
  topic.sendEvent("init_pre",evtMap);
}

/**
 * Method:  init[]
 */
void
functions::PiFunction_impl::init_impl (
  /* in array<string> */::sidl::array< ::std::string>& params ) 
{
  // DO-NOT-DELETE splicer.begin(functions.PiFunction.init)
  // Insert-Code-Here {functions.PiFunction.init} (init method)
  // DO-NOT-DELETE splicer.end(functions.PiFunction.init)
}

/**
 * Method:  init_post[]
 */
void
functions::PiFunction_impl::init_post_impl (
  /* in array<string> */::sidl::array< ::std::string>& params ) 
{
  gov::cca::TypeMap evtMap = typemap.cloneEmpty();
  evtMap.putStringArray("params",params);
  topic.sendEvent("init_post",evtMap);
}

/**
 * Method:  evaluate_pre[]
 */
void
functions::PiFunction_impl::evaluate_pre_impl (
  /* in */double x ) 
{
  gov::cca::TypeMap evtMap = typemap.cloneEmpty();
  evtMap.putDouble("x",x);
  topic.sendEvent("evaluate_pre",evtMap);
}

/**
 * Method:  evaluate[]
 */
double
functions::PiFunction_impl::evaluate_impl (
  /* in */double x ) 
{
  // DO-NOT-DELETE splicer.begin(functions.PiFunction.evaluate)
  // Insert-Code-Here {functions.PiFunction.evaluate} (evaluate method)

  return 4.0 / (1.0 + x * x);

  // DO-NOT-DELETE splicer.end(functions.PiFunction.evaluate)
}

/**
 * Method:  evaluate_post[]
 */
void
functions::PiFunction_impl::evaluate_post_impl (
  /* in */double x,
  /* in */double _retval ) 
{
  gov::cca::TypeMap evtMap = typemap.cloneEmpty();
  evtMap.putDouble("x",x);
  evtMap.putDouble("_retval",_retval);
  topic.sendEvent("evaluate_post",evtMap);
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
functions::PiFunction_impl::setServices_pre_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  printf("setServices_pre is called\n");
  services.registerUsesPort("evtSvcPublisher","cca.EventService",0);
  gov::cca::Port evtPort = services.getPort("evtSvcPublisher");
  sci::cca::ports::PublisherEventService eventSvc = 
    babel_cast<sci::cca::ports::PublisherEventService>(evtPort);
  if(eventSvc._is_nil()) {
    std::cerr << "Unable to get event service from framework\n";
    exit(1);
  }
  topic = eventSvc.getTopic("log.all");
  typemap = services.createTypeMap();
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
functions::PiFunction_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(functions.PiFunction.setServices)
  // Insert-Code-Here {functions.PiFunction.setServices} (setServices method)

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
    
    services.addProvidesPort(p,
                             "FunctionPort-pp",
                             "function.FunctionPort", tm);
  }

  // DO-NOT-DELETE splicer.end(functions.PiFunction.setServices)
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
functions::PiFunction_impl::setServices_post_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(functions.PiFunction.setServices_post)
  // Insert-Code-Here {functions.PiFunction.setServices_post} (setServices_post)
  // DO-NOT-DELETE splicer.end(functions.PiFunction.setServices_post)
}


// DO-NOT-DELETE splicer.begin(functions.PiFunction._misc)
// Insert-Code-Here {functions.PiFunction._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(functions.PiFunction._misc)

