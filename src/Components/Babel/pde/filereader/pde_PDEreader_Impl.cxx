// 
// File:          pde_PDEreader_Impl.cxx
// Symbol:        pde.PDEreader-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.PDEreader
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "pde_PDEreader_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(pde.PDEreader._includes)
#include <fstream>
#define L_INPUT_FILE "L.pde"
// DO-NOT-DELETE splicer.end(pde.PDEreader._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
pde::PDEreader_impl::PDEreader_impl() : StubBase(reinterpret_cast< void*>(
  ::pde::PDEreader::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(pde.PDEreader._ctor2)
  // Insert-Code-Here {pde.PDEreader._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(pde.PDEreader._ctor2)
}

// user defined constructor
void pde::PDEreader_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(pde.PDEreader._ctor)
  // Insert-Code-Here {pde.PDEreader._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(pde.PDEreader._ctor)
}

// user defined destructor
void pde::PDEreader_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(pde.PDEreader._dtor)
  // Insert-Code-Here {pde.PDEreader._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(pde.PDEreader._dtor)
}

// static class initializer
void pde::PDEreader_impl::_load() {
  // DO-NOT-DELETE splicer.begin(pde.PDEreader._load)
  // Insert-Code-Here {pde.PDEreader._load} (class initialization)
  // DO-NOT-DELETE splicer.end(pde.PDEreader._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getPDEdescription[]
 */
int32_t
pde::PDEreader_impl::getPDEdescription_impl (
  /* out array<double> */::sidl::array<double>& nodes,
  /* out array<int> */::sidl::array<int32_t>& boundaries,
  /* out array<int> */::sidl::array<int32_t>& dirichletNodes,
  /* out array<double> */::sidl::array<double>& dirichletValues ) 
{
  // DO-NOT-DELETE splicer.begin(pde.PDEreader.getPDEdescription)
  std::ifstream is(L_INPUT_FILE);
 
  while(true) {
    std::string name;
    is >> name;
    if (name == "node") {
      int cnt;
      is >> cnt;
      nodes = sidl::array<double>::create1d(cnt*2);  
      for (int i = 0, j = 0; i < cnt; i++, j+=2) {
        double x, y;
        is >> x >> y;
        nodes.set(j,x);
        nodes.set(j+1,y);
      }
    } else if(name == "boundary") {
      int cnt;
      is >> cnt;
      boundaries = sidl::array<int>::create1d(cnt);  
      for (int i = 0; i < cnt; i++) {
        int index;
        is >> index;
        boundaries.set(i,index);
      }
    } else if (name == "dirichlet") {
      int cnt;
      is >> cnt;
      dirichletNodes = sidl::array<int>::create1d(cnt);  
      dirichletValues = sidl::array<double>::create1d(cnt);
      for (int i = 0; i < cnt; i++) {
        int index;
        is >> index;
        dirichletNodes.set(i,index);
      }
      for (int i = 0; i < cnt; i++) {
        double value;
        is >> value;
        dirichletValues.set(i,value);
      }
    } else if (name == "end") {
      break;
    }
  }
  return 0;
  // DO-NOT-DELETE splicer.end(pde.PDEreader.getPDEdescription)
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
pde::PDEreader_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(pde.PDEreader.setServices)
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

    services.addProvidesPort(p,"pde","pdeports.PDEdescriptionPort",tm);
  }
  // DO-NOT-DELETE splicer.end(pde.PDEreader.setServices)
}


// DO-NOT-DELETE splicer.begin(pde.PDEreader._misc)
// Insert-Code-Here {pde.PDEreader._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(pde.PDEreader._misc)

