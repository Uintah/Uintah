// 
// File:          pde_PDEdriver_Skel.cxx
// Symbol:        pde.PDEdriver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side glue code for pde.PDEdriver
// 
// WARNING: Automatically generated; changes will be lost
// 
// 
#ifndef included_pde_PDEdriver_Impl_hxx
#include "pde_PDEdriver_Impl.hxx"
#endif
#ifndef included_pde_PDEdriver_IOR_h
#include "pde_PDEdriver_IOR.h"
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

  static void
  skel_pde_PDEdriver_setServices( 
  /* in */ struct pde_PDEdriver__object* self, /* in */ struct 
    gov_cca_Services__object* services, sidl_BaseInterface__object ** 
    _exception )
  {
    // pack args to dispatch to impl
    ::pde::PDEdriver_impl *_this = reinterpret_cast< ::pde::PDEdriver_impl*>(
      self->d_data);
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

  static int32_t
  skel_pde_PDEdriver_go( 
  /* in */ struct pde_PDEdriver__object* self, sidl_BaseInterface__object ** 
    _exception )
  {
    // pack args to dispatch to impl
    int32_t _result = 0;
    ::pde::PDEdriver_impl *_this = reinterpret_cast< ::pde::PDEdriver_impl*>(
      self->d_data);
    *_exception = 0;//init just to be safe

    // dispatch to impl
    try { 
      _result = _this->go_impl(  );
    } catch ( ::sidl::RuntimeException _ex ) { 
      sidl_BaseInterface _throwaway_exception;
      (_ex._get_ior()->d_epv->f_add) (_ex._get_ior()->d_object,  __FILE__, __LINE__,"C++ skeleton for go.", &_throwaway_exception);
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

    return _result;
  }

#ifdef WITH_RMI
  struct gov_cca_Services__object* skel_pde_PDEdriver_fconnect_gov_cca_Services(
    const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
    return gov_cca_Services__connectI(url, ar, _ex);
  }

  struct sidl_BaseInterface__object* 
    skel_pde_PDEdriver_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
    ar, sidl_BaseInterface *_ex) { 
    return sidl_BaseInterface__connectI(url, ar, _ex);
  }

#endif /*WITH_RMI*/
  void
  pde_PDEdriver__call_load(void) {::pde::PDEdriver_impl::_load();}

  static void
  skel_pde_PDEdriver__ctor(struct pde_PDEdriver__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      self->d_data = reinterpret_cast< void*>(new ::pde::PDEdriver_impl(self));
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
  skel_pde_PDEdriver__ctor2(struct pde_PDEdriver__object* self, void* 
    private_data, struct sidl_BaseInterface__object **_ex ) { 
    *_ex = NULL;
  }

  static void
  skel_pde_PDEdriver__dtor(struct pde_PDEdriver__object* self, struct 
    sidl_BaseInterface__object **_ex ) { 
    try {
      ::pde::PDEdriver_impl* loc_data = reinterpret_cast< 
        ::pde::PDEdriver_impl*>(self->d_data);
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
  pde_PDEdriver__set_epv(struct pde_PDEdriver__epv *epv,
  struct pde_PDEdriver__pre_epv *pre_epv, struct pde_PDEdriver__post_epv *post_epv){
    // initialize builtin methods
    epv->f__ctor = skel_pde_PDEdriver__ctor;
    epv->f__ctor2 = skel_pde_PDEdriver__ctor2;
    epv->f__dtor = skel_pde_PDEdriver__dtor;
    // initialize local methods
    pre_epv->f_setServices_pre = NULL;
    epv->f_setServices = skel_pde_PDEdriver_setServices;
    post_epv->f_setServices_post = NULL;
    pre_epv->f_go_pre = NULL;
    epv->f_go = skel_pde_PDEdriver_go;
    post_epv->f_go_post = NULL;
  }


} // end extern "C"
