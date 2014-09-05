// 
// File:          pde_Tri_Impl.cxx
// Symbol:        pde.Tri-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.Tri
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "pde_Tri_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(pde.Tri._includes)
#include "Delaunay.h"

using namespace SCIRun;
using namespace Tri;
// DO-NOT-DELETE splicer.end(pde.Tri._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
pde::Tri_impl::Tri_impl() : StubBase(reinterpret_cast< void*>(
  ::pde::Tri::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(pde.Tri._ctor2)
  // Insert-Code-Here {pde.Tri._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(pde.Tri._ctor2)
}

// user defined constructor
void pde::Tri_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(pde.Tri._ctor)
  // Insert-Code-Here {pde.Tri._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(pde.Tri._ctor)
}

// user defined destructor
void pde::Tri_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(pde.Tri._dtor)
  // Insert-Code-Here {pde.Tri._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(pde.Tri._dtor)
}

// static class initializer
void pde::Tri_impl::_load() {
  // DO-NOT-DELETE splicer.begin(pde.Tri._load)
  // Insert-Code-Here {pde.Tri._load} (class initialization)
  // DO-NOT-DELETE splicer.end(pde.Tri._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  triangulate[]
 */
int32_t
pde::Tri_impl::triangulate_impl (
  /* in array<double> */::sidl::array<double>& nodes,
  /* in array<int> */::sidl::array<int32_t>& boundaries,
  /* out array<int> */::sidl::array<int32_t>& triangles ) 
{
  // DO-NOT-DELETE splicer.begin(pde.Tri.triangulate)
  Delaunay* mesh = new Delaunay(nodes, boundaries);
  mesh->triangulation();

  std::vector<Triangle> tri = mesh->getTriangles();
  triangles = sidl::array<int>::create1d(tri.size()*3);

  for (unsigned int i = 0, j = 0; i < tri.size(); i++, j+=3) {
    triangles.set(j,tri[i].index[0] - 4);
    triangles.set(j+1,tri[i].index[1] - 4);
    triangles.set(j+2,tri[i].index[2] - 4);
  }
  delete mesh;
  return 0;
  // DO-NOT-DELETE splicer.end(pde.Tri.triangulate)
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
pde::Tri_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(pde.Tri.setServices)
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

    services.addProvidesPort(p,"mesh","pdeports.MeshPort",tm);
  }
  // DO-NOT-DELETE splicer.end(pde.Tri.setServices)
}


// DO-NOT-DELETE splicer.begin(pde.Tri._misc)
// Insert-Code-Here {pde.Tri._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(pde.Tri._misc)

