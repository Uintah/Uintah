// 
// File:          pde_FEM_Skel.cxx
// Symbol:        pde.FEM-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side glue code for pde.FEM
// 
// WARNING: Automatically generated; changes will be lost
// 
// 
#ifndef included_pde_FEM_Impl_hxx
#include "pde_FEM_Impl.hxx"
#endif
#ifndef included_pde_FEM_IOR_h
#include "pde_FEM_IOR.h"
#endif
#ifndef included_sidl_BaseException_hxx
#include "sidl_BaseException.hxx"
#endif
#ifndef included_sidl_LangSpecificException_hxx
#include "sidl_LangSpecificException.hxx"
#endif
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
#include "sidl_String.h"
#include <stddef.h>
#include <cstring>

extern "C" {

  static int32_t
  skel_pde_FEM_makeFEMmatrices( 
  /* in */ struct pde_FEM__object* self, /* in array<int> */ struct 
    sidl_int__array* mesh,/* in array<double> */ struct sidl_double__array* 
    nodes,/* in array<int> */ struct sidl_int__array* dirichletNodes,/* in 
    array<double> */ struct sidl_double__array* dirichletValues,/* out 
    array<double,2> */ struct sidl_double__array** Ag,/* out array<double> */ 
    struct sidl_double__array** fg,/* out */ int32_t* size, 
    sidl_BaseInterface__object ** _exception )
  {
    // pack args to dispatch to impl
    int32_t _result = 0;
    ::pde::FEM_impl *_this = reinterpret_cast< ::pde::FEM_impl*>(self->d_data);
    ::sidl::array<int32_t> _local_mesh(mesh);
    if (mesh) {
      _local_mesh.addRef();
     }
    ::sidl::array<double> _local_nodes(nodes);
    if (nodes) {
      _local_nodes.addRef();
     }
    ::sidl::array<int32_t> _local_dirichletNodes(dirichletNodes);
    if (dirichletNodes) {
      _local_dirichletNodes.addRef();
     }
    ::sidl::array<double> _local_dirichletValues(dirichletValues);
    if (dirichletValues) {
      _local_dirichletValues.addRef();
     }
    ::sidl::array<double> _local_Ag;
    ::sidl::array<double> _local_fg;
    *_exception = 0;//init just to be safe

    // dispatch to impl
    try { 
      _result = _this->makeFEMmatrices_impl( /* in array<int> */ _local_mesh, 
        /* in array<double> */ _local_nodes, /* in array<int> */ 
        _local_dirichletNodes, /* in array<double> */ _local_dirichletValues, 
        /* out array<double,2> */ _local_Ag, /* out array<double> */ _local_fg, 
        /* out */ *size );
    } catch ( ::sidl::RuntimeException _ex ) { 
      sidl_BaseInterface _throwaway_exception;
      (_ex._get_ior()->d_epv->f_add) (_ex._get_ior()->d_object,  __FILE__, __LINE__,"C++ skeleton for makeFEMmatrices.", &_throwaway_exception);
      _ex.addRef();
      *_exception = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _ex._get_ior());

      return _result;
    } catch (...) {
      // Convert all other exceptions to LangSpecific
      ::sidl::LangSpecificException _unexpected = 
        ::sidl::LangSpecificException::_create();
      sidl_BaseInterface _throwaway_exception;
      (_unexpected._get_ior()->d_epv->f_setNote) (_unexpected._get_ior(),
        "Unexpected C++ exception", &_throwaway_exception);
      _unexpected.addRef();
      *_exception = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _unexpected._get_ior());

      return _result;
    }
    // unpack results and cleanup
    if ( _local_Ag._not_nil() ) {
      _local_Ag.addRef();
    }
    *Ag = _local_Ag._get_ior();
    if ( _local_fg._not_nil() ) {
      _local_fg.addRef();
    }
    *fg = _local_fg._get_ior();
    return _result;
  }

  static void
  skel_pde_FEM_setServices( 
  /* in */ struct pde_FEM__object* self, /* in */ struct 
    gov_cca_Services__object* services, sidl_BaseInterface__object ** 
    _exception )
  {
    // pack args to dispatch to impl
    ::pde::FEM_impl *_this = reinterpret_cast< ::pde::FEM_impl*>(self->d_data);
    ::gov::cca::Services  _local_services(services);
    *_exception = 0;//init just to be safe

    // dispatch to impl
    try { 
      _this->setServices_impl( /* in */ _local_services );
    } catch ( ::sidl::BaseException _ex ) { 
      sidl_BaseInterface _throwaway_exception;
      (_ex._get_ior()->d_epv->f_add) (_ex._get_ior()->d_object,  __FILE__, __LINE__,"C++ skeleton for setServices.", &_throwaway_exception);
      _ex.addRef();
      *_exception = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _ex._get_ior());

    } catch (...) {
      // Convert all other exceptions to LangSpecific
      ::sidl::LangSpecificException _unexpected = 
        ::sidl::LangSpecificException::_create();
      sidl_BaseInterface _throwaway_exception;
      (_unexpected._get_ior()->d_epv->f_setNote) (_unexpected._get_ior(),
        "Unexpected C++ exception", &_throwaway_exception);
      _unexpected.addRef();
      *_exception = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _unexpected._get_ior());

    }
    // unpack results and cleanup

  }

#ifdef WITH_RMI
  struct gov_cca_Services__object* skel_pde_FEM_fconnect_gov_cca_Services(const 
    char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
    return gov_cca_Services__connectI(url, ar, _ex);
  }

  struct sidl_BaseInterface__object* skel_pde_FEM_fconnect_sidl_BaseInterface(
    const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
    return sidl_BaseInterface__connectI(url, ar, _ex);
  }

#endif /*WITH_RMI*/
  void
  pde_FEM__call_load(void) {::pde::FEM_impl::_load();}

  static void
  skel_pde_FEM__ctor(struct pde_FEM__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      self->d_data = reinterpret_cast< void*>(new ::pde::FEM_impl(self));
    } catch ( ::sidl::RuntimeException _exc ) {
      sidl_BaseInterface _throwaway_exception;
      (_exc._get_ior()->d_epv->f_add) (_exc._get_ior()->d_object, __FILE__, __LINE__," C++ skeleton for _ctor.", &_throwaway_exception);
      _exc.addRef();
      *_ex = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _exc._get_ior());
    } catch (...) {
      // Convert all other exceptions to LangSpecific
      ::sidl::LangSpecificException _unexpected = 
        ::sidl::LangSpecificException::_create();
      _unexpected.setNote("Unexpected C++ exception in _ctor.");
      _unexpected.addRef();
      *_ex = reinterpret_cast< struct sidl_BaseInterface__object * >(
        _unexpected._get_ior());
    }
  }

  static void
  skel_pde_FEM__ctor2(struct pde_FEM__object* self, void* private_data, struct 
    sidl_BaseInterface__object **_ex ) { 
    *_ex = NULL;
  }

  static void
  skel_pde_FEM__dtor(struct pde_FEM__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      ::pde::FEM_impl* loc_data = reinterpret_cast< ::pde::FEM_impl*>(
        self->d_data);
      if(!loc_data->_isWrapped()) {
        delete (loc_data);
      }
    } catch ( ::sidl::RuntimeException _exc ) {
      sidl_BaseInterface _throwaway_exception;
      (_exc._get_ior()->d_epv->f_add) (_exc._get_ior()->d_object, __FILE__, __LINE__," C++ skeleton for _dtor.", &_throwaway_exception);
      _exc.addRef();
      *_ex = reinterpret_cast< struct sidl_BaseInterface__object * >(_exc._get_ior());
    } catch (...) {
      // Convert all other exceptions to LangSpecific
      ::sidl::LangSpecificException _unexpected = ::sidl::LangSpecificException::_create();
      _unexpected.setNote("Unexpected C++ exception in _dtor.");
      _unexpected.addRef();
      *_ex = reinterpret_cast< struct sidl_BaseInterface__object * >(_unexpected._get_ior());
    }
  }

  void
  pde_FEM__set_epv(struct pde_FEM__epv *epv,
  struct pde_FEM__pre_epv *pre_epv, struct pde_FEM__post_epv *post_epv){
    // initialize builtin methods
    epv->f__ctor = skel_pde_FEM__ctor;
    epv->f__ctor2 = skel_pde_FEM__ctor2;
    epv->f__dtor = skel_pde_FEM__dtor;
    // initialize local methods
    pre_epv->f_makeFEMmatrices_pre = NULL;
    epv->f_makeFEMmatrices = skel_pde_FEM_makeFEMmatrices;
    post_epv->f_makeFEMmatrices_post = NULL;
    pre_epv->f_setServices_pre = NULL;
    epv->f_setServices = skel_pde_FEM_setServices;
    post_epv->f_setServices_post = NULL;
  }


} // end extern "C"
