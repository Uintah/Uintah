// 
// File:          pde_FEM_Impl.cxx
// Symbol:        pde.FEM-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.FEM
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "pde_FEM_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(pde.FEM._includes)
#include "FEM.h"
#include <iostream>

using namespace SCIRun;
// DO-NOT-DELETE splicer.end(pde.FEM._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
pde::FEM_impl::FEM_impl() : StubBase(reinterpret_cast< void*>(
  ::pde::FEM::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(pde.FEM._ctor2)
  // Insert-Code-Here {pde.FEM._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(pde.FEM._ctor2)
}

// user defined constructor
void pde::FEM_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(pde.FEM._ctor)
  // Insert-Code-Here {pde.FEM._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(pde.FEM._ctor)
}

// user defined destructor
void pde::FEM_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(pde.FEM._dtor)
  // Insert-Code-Here {pde.FEM._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(pde.FEM._dtor)
}

// static class initializer
void pde::FEM_impl::_load() {
  // DO-NOT-DELETE splicer.begin(pde.FEM._load)
  // Insert-Code-Here {pde.FEM._load} (class initialization)
  // DO-NOT-DELETE splicer.end(pde.FEM._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  makeFEMmatrices[]
 */
int32_t
pde::FEM_impl::makeFEMmatrices_impl (
  /* in array<int> */::sidl::array<int32_t>& mesh,
  /* in array<double> */::sidl::array<double>& nodes,
  /* in array<int> */::sidl::array<int32_t>& dirichletNodes,
  /* in array<double> */::sidl::array<double>& dirichletValues,
  /* out array<double,2> */::sidl::array<double>& Ag,
  /* out array<double> */::sidl::array<double>& fg,
  /* out */int32_t& size ) 
{
  // DO-NOT-DELETE splicer.begin(pde.FEM.makeFEMmatrices)
  FEMgenerator fem(dirichletNodes, dirichletValues);
  if (nodes.length() == 0  || mesh.length() == 0) {
    std::cerr << "FEMgenerator: Bad mesh or nodes!" << std::endl;
    return 1;
  } 
  
  fem.globalMatrices(nodes,mesh);
  Ag = fem.Ag;
  fg = fem.fg;
  size = fem.Ag.length();

  return 0;
  // DO-NOT-DELETE splicer.end(pde.FEM.makeFEMmatrices)
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
pde::FEM_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(pde.FEM.setServices)
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

    services.addProvidesPort(p,"fem_matrix","pdeports.FEMmatrixPort",tm);
  }
  // DO-NOT-DELETE splicer.end(pde.FEM.setServices)
}


// DO-NOT-DELETE splicer.begin(pde.FEM._misc)
// Insert-Code-Here {pde.FEM._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(pde.FEM._misc)

