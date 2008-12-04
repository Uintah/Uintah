// 
// File:          pde_Tri_Skel.cxx
// Symbol:        pde.Tri-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side glue code for pde.Tri
// 
// WARNING: Automatically generated; changes will be lost
// 
// 
#ifndef included_pde_Tri_Impl_hxx
#include "pde_Tri_Impl.hxx"
#endif
#ifndef included_pde_Tri_IOR_h
#include "pde_Tri_IOR.h"
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
  skel_pde_Tri_triangulate( 
  /* in */ struct pde_Tri__object* self, /* in array<double> */ struct 
    sidl_double__array* nodes,/* in array<int> */ struct sidl_int__array* 
    boundaries,/* out array<int> */ struct sidl_int__array** triangles, 
    sidl_BaseInterface__object ** _exception )
  {
    // pack args to dispatch to impl
    int32_t _result = 0;
    ::pde::Tri_impl *_this = reinterpret_cast< ::pde::Tri_impl*>(self->d_data);
    ::sidl::array<double> _local_nodes(nodes);
    if (nodes) {
      _local_nodes.addRef();
     }
    ::sidl::array<int32_t> _local_boundaries(boundaries);
    if (boundaries) {
      _local_boundaries.addRef();
     }
    ::sidl::array<int32_t> _local_triangles;
    *_exception = 0;//init just to be safe

    // dispatch to impl
    try { 
      _result = _this->triangulate_impl( /* in array<double> */ _local_nodes, 
        /* in array<int> */ _local_boundaries, /* out array<int> */ 
        _local_triangles );
    } catch ( ::sidl::RuntimeException _ex ) { 
      sidl_BaseInterface _throwaway_exception;
      (_ex._get_ior()->d_epv->f_add) (_ex._get_ior()->d_object,  __FILE__, __LINE__,"C++ skeleton for triangulate.", &_throwaway_exception);
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
    if ( _local_triangles._not_nil() ) {
      _local_triangles.addRef();
    }
    *triangles = _local_triangles._get_ior();
    return _result;
  }

  static void
  skel_pde_Tri_setServices( 
  /* in */ struct pde_Tri__object* self, /* in */ struct 
    gov_cca_Services__object* services, sidl_BaseInterface__object ** 
    _exception )
  {
    // pack args to dispatch to impl
    ::pde::Tri_impl *_this = reinterpret_cast< ::pde::Tri_impl*>(self->d_data);
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
  struct gov_cca_Services__object* skel_pde_Tri_fconnect_gov_cca_Services(const 
    char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
    return gov_cca_Services__connectI(url, ar, _ex);
  }

  struct sidl_BaseInterface__object* skel_pde_Tri_fconnect_sidl_BaseInterface(
    const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
    return sidl_BaseInterface__connectI(url, ar, _ex);
  }

#endif /*WITH_RMI*/
  void
  pde_Tri__call_load(void) {::pde::Tri_impl::_load();}

  static void
  skel_pde_Tri__ctor(struct pde_Tri__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      self->d_data = reinterpret_cast< void*>(new ::pde::Tri_impl(self));
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
  skel_pde_Tri__ctor2(struct pde_Tri__object* self, void* private_data, struct 
    sidl_BaseInterface__object **_ex ) { 
    *_ex = NULL;
  }

  static void
  skel_pde_Tri__dtor(struct pde_Tri__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      ::pde::Tri_impl* loc_data = reinterpret_cast< ::pde::Tri_impl*>(
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
  pde_Tri__set_epv(struct pde_Tri__epv *epv,
  struct pde_Tri__pre_epv *pre_epv, struct pde_Tri__post_epv *post_epv){
    // initialize builtin methods
    epv->f__ctor = skel_pde_Tri__ctor;
    epv->f__ctor2 = skel_pde_Tri__ctor2;
    epv->f__dtor = skel_pde_Tri__dtor;
    // initialize local methods
    pre_epv->f_triangulate_pre = NULL;
    epv->f_triangulate = skel_pde_Tri_triangulate;
    post_epv->f_triangulate_post = NULL;
    pre_epv->f_setServices_pre = NULL;
    epv->f_setServices = skel_pde_Tri_setServices;
    post_epv->f_setServices_post = NULL;
  }


} // end extern "C"
