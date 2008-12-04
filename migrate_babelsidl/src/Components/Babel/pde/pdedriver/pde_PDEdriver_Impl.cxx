// 
// File:          pde_PDEdriver_Impl.cxx
// Symbol:        pde.PDEdriver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.PDEdriver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "pde_PDEdriver_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(pde.PDEdriver._includes)
// Insert-Code-Here {pde.PDEdriver._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(pde.PDEdriver._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
pde::PDEdriver_impl::PDEdriver_impl() : StubBase(reinterpret_cast< void*>(
  ::pde::PDEdriver::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver._ctor2)
  // Insert-Code-Here {pde.PDEdriver._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(pde.PDEdriver._ctor2)
}

// user defined constructor
void pde::PDEdriver_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver._ctor)
  // Insert-Code-Here {pde.PDEdriver._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(pde.PDEdriver._ctor)
}

// user defined destructor
void pde::PDEdriver_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver._dtor)
  // Insert-Code-Here {pde.PDEdriver._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(pde.PDEdriver._dtor)
}

// static class initializer
void pde::PDEdriver_impl::_load() {
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver._load)
  // Insert-Code-Here {pde.PDEdriver._load} (class initialization)
  // DO-NOT-DELETE splicer.end(pde.PDEdriver._load)
}

// user defined static methods: (none)

// user defined non-static methods:
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
pde::PDEdriver_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver.setServices)
  if(services._not_nil()) {
    
    this->services = services;

    gov::cca::Port gp = (*this);      //  Babel required casting
    
    if(gp._is_nil()) {
      fprintf(stderr, "gp is nil");
    } 
    services.addProvidesPort(gp,"goport","gov.cca.ports.GoPort",services.createTypeMap());

    services.registerUsesPort("pde","pdeports.PDEdescriptionPort", services.createTypeMap());
    services.registerUsesPort("mesh","pdeports.MeshPort", services.createTypeMap());
    services.registerUsesPort("fem_matrix","pdeports.FEMmatrixPort", services.createTypeMap());
    services.registerUsesPort("linsolver","pdeports.LinSolverPort", services.createTypeMap());
    services.registerUsesPort("viewer","pdeports.ViewPort", services.createTypeMap());
  }
  // DO-NOT-DELETE splicer.end(pde.PDEdriver.setServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
pde::PDEdriver_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver.go)
  sidl::array<double> nodes = NULL;  
  sidl::array<int> boundries = NULL;  
  sidl::array<int> dirichletNodes = NULL;  
  sidl::array<double> dirichletValues = NULL;
  sidl::array<int> triangles = NULL;
  sidl::array<double> Ag = NULL;
  sidl::array<double> fg = NULL;
  sidl::array<double> x = NULL;
  int size;

  pdeports::PDEdescriptionPort pdePort;
  gov::cca::Port pp = services.getPort("pde");
  pdePort = babel_cast<pdeports::PDEdescriptionPort>(pp);
  if(pdePort._is_nil()) {
    fprintf(stdout, "pdereader not connected\n");
    services.releasePort("pde");
    return -1;   
  }
  pdePort.getPDEdescription(nodes, boundries, dirichletNodes, dirichletValues);
 
  pdeports::MeshPort meshPort;
  gov::cca::Port mp = services.getPort("mesh");
  meshPort = babel_cast<pdeports::MeshPort>(mp);
  if(meshPort._is_nil()) {
    fprintf(stdout, "meshport not connected\n");
    services.releasePort("mesh");
    return -1;   
  }
  meshPort.triangulate(nodes, boundries, triangles);
  
  pdeports::FEMmatrixPort fem_matrixPort;
  gov::cca::Port femp = services.getPort("fem_matrix");
  fem_matrixPort = babel_cast<pdeports::FEMmatrixPort>(femp);
  if(fem_matrixPort._is_nil()) {
    fprintf(stdout, "fem_matrixport not connected\n");
    services.releasePort("fem_matrix");
    return -1;   
  }
  fem_matrixPort.makeFEMmatrices(triangles, nodes,
                                  dirichletNodes, dirichletValues,
                                  Ag, fg, size);

  pdeports::LinSolverPort linsolverport;
  gov::cca::Port lsp = services.getPort("linsolver");
  linsolverport = babel_cast<pdeports::LinSolverPort>(lsp);
  if(linsolverport._is_nil()) {
    fprintf(stdout, "linsolverport not connected\n");
    services.releasePort("linsolver");
    return -1;   
  }
  linsolverport.jacobi(Ag, fg, x);

  services.releasePort("fem_matrix");
  services.releasePort("mesh");
  services.releasePort("pde");
  services.releasePort("linsolver");
  // DO-NOT-DELETE splicer.end(pde.PDEdriver.go)
}


// DO-NOT-DELETE splicer.begin(pde.PDEdriver._misc)
// Insert-Code-Here {pde.PDEdriver._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(pde.PDEdriver._misc)

