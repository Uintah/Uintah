// 
// File:          pde_LinSolver_Impl.cxx
// Symbol:        pde.LinSolver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.LinSolver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "pde_LinSolver_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(pde.LinSolver._includes)
#include <iostream>
// DO-NOT-DELETE splicer.end(pde.LinSolver._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
pde::LinSolver_impl::LinSolver_impl() : StubBase(reinterpret_cast< void*>(
  ::pde::LinSolver::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(pde.LinSolver._ctor2)
  // Insert-Code-Here {pde.LinSolver._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(pde.LinSolver._ctor2)
}

// user defined constructor
void pde::LinSolver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(pde.LinSolver._ctor)
  // Insert-Code-Here {pde.LinSolver._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(pde.LinSolver._ctor)
}

// user defined destructor
void pde::LinSolver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(pde.LinSolver._dtor)
  // Insert-Code-Here {pde.LinSolver._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(pde.LinSolver._dtor)
}

// static class initializer
void pde::LinSolver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(pde.LinSolver._load)
  // Insert-Code-Here {pde.LinSolver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(pde.LinSolver._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  jacobi[]
 */
int32_t
pde::LinSolver_impl::jacobi_impl (
  /* in array<double,2> */::sidl::array<double>& A,
  /* in array<double> */::sidl::array<double>& b,
  /* out array<double> */::sidl::array<double>& x ) 
{
  // DO-NOT-DELETE splicer.begin(pde.LinSolver.jacobi)
  const double eps = 1e-10;
  const int maxiter = 10000;

  int N = b.length();
  x = sidl::array<double>::create1d(N);
  for(int i=0; i < x.length(); i++) {
    x.set(i,1.0);
  }

  int iter;

  for (iter = 0; iter < maxiter; iter++) {
    double norm = 0;
    for (int i = 0; i < N; i++) {
      double res_i = 0;
      for (int k = 0; k < N; k++) {
        res_i += A.get(i,k) * x.get(k);
      }
      res_i -= b.get(i);
      norm += res_i * res_i;
    }
    if (norm < eps * eps) break;
    std::cerr<<"iter = "<<iter<<"  norm2 = "<<norm<<std::endl;

    sidl::array<double> tempx = sidl::array<double>::create1d(N);
    tempx.copy(x);
    for (int i = 0; i < N; i++) {
      tempx.set(i, b.get(i));
      for (int k = 0; k < N; k++) {
        if (i == k) continue;
        tempx.set(i, tempx.get(i) - A.get(i,k)*x.get(k));
      }
      tempx.set(i, tempx.get(i) / A.get(i,i));
    }
    x.copy(tempx);
  }

  if (iter != maxiter) {
    return 0;
  } else {
    return 1;
  }
  // DO-NOT-DELETE splicer.end(pde.LinSolver.jacobi)
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
pde::LinSolver_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(pde.LinSolver.setServices)
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

    services.addProvidesPort(p,"linsolver","pdeports.LinSolverPort",tm);
  }
  // DO-NOT-DELETE splicer.end(pde.LinSolver.setServices)
}


// DO-NOT-DELETE splicer.begin(pde.LinSolver._misc)
// Insert-Code-Here {pde.LinSolver._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(pde.LinSolver._misc)

