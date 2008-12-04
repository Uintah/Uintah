/*
 * File:          pde_PDEdriver_IOR.h
 * Symbol:        pde.PDEdriver-v0.1
 * Symbol Type:   class
 * Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
 * Description:   Intermediate Object Representation for pde.PDEdriver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_pde_PDEdriver_IOR_h
#define included_pde_PDEdriver_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_gov_cca_Component_IOR_h
#include "gov_cca_Component_IOR.h"
#endif
#ifndef included_gov_cca_Port_IOR_h
#include "gov_cca_Port_IOR.h"
#endif
#ifndef included_gov_cca_ports_GoPort_IOR_h
#include "gov_cca_ports_GoPort_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "pde.PDEdriver" (version 0.1)
 */

struct pde_PDEdriver__array;
struct pde_PDEdriver__object;

/*
 * Forward references for external classes and interfaces.
 */

struct gov_cca_CCAException__array;
struct gov_cca_CCAException__object;
struct gov_cca_Services__array;
struct gov_cca_Services__object;
struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct pde_PDEdriver__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 9 */
  void (*f__ctor)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 10 */
  void (*f__ctor2)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 11 */
  void (*f__dtor)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 12 */
  void (*f__load)(
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.17 */
  /* Methods introduced in gov.cca.Component-v0.8.1 */
  void (*f_setServices)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ struct gov_cca_Services__object* services,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in gov.cca.Port-v0.8.1 */
  /* Methods introduced in gov.cca.ports.GoPort-v0.8.1 */
  int32_t (*f_go)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in pde.PDEdriver-v0.1 */
};

/*
 * Declare the method pre hooks entry point vector.
 */

struct pde_PDEdriver__pre_epv {
  void (*f_setServices_pre)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ struct gov_cca_Services__object* services,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_go_pre)(
    /* in */ struct pde_PDEdriver__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method post hooks entry point vector.
 */

struct pde_PDEdriver__post_epv {
  void (*f_setServices_post)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ struct gov_cca_Services__object* services,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_go_post)(
    /* in */ struct pde_PDEdriver__object* self,
    /* in */ int32_t _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Define the controls and statistics structure.
 */


struct pde_PDEdriver__cstats {
  sidl_bool use_hooks;
};

/*
 * Define the class object structure.
 */

struct pde_PDEdriver__object {
  struct sidl_BaseClass__object       d_sidl_baseclass;
  struct gov_cca_Component__object    d_gov_cca_component;
  struct gov_cca_Port__object         d_gov_cca_port;
  struct gov_cca_ports_GoPort__object d_gov_cca_ports_goport;
  struct pde_PDEdriver__epv*          d_epv;
  struct pde_PDEdriver__cstats        d_cstats;
  void*                               d_data;
};

struct pde_PDEdriver__external {
  struct pde_PDEdriver__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct pde_PDEdriver__external*
pde_PDEdriver__externals(void);

extern struct pde_PDEdriver__object*
pde_PDEdriver__createObject(void* ddata,struct sidl_BaseInterface__object ** 
  _ex);

extern void pde_PDEdriver__init(
  struct pde_PDEdriver__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);

extern void pde_PDEdriver__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct gov_cca_Component__epv **s_arg_epv__gov_cca_component,
  struct gov_cca_Component__epv **s_arg_epv_hooks__gov_cca_component,
  struct gov_cca_Port__epv **s_arg_epv__gov_cca_port,
  struct gov_cca_Port__epv **s_arg_epv_hooks__gov_cca_port,
  struct gov_cca_ports_GoPort__epv **s_arg_epv__gov_cca_ports_goport,
  struct gov_cca_ports_GoPort__epv **s_arg_epv_hooks__gov_cca_ports_goport,
  struct pde_PDEdriver__epv **s_arg_epv__pde_pdedriver,
  struct pde_PDEdriver__epv **s_arg_epv_hooks__pde_pdedriver);

extern void pde_PDEdriver__fini(
  struct pde_PDEdriver__object* self, struct sidl_BaseInterface__object ** _ex);

extern void pde_PDEdriver__IOR_version(int32_t *major, int32_t *minor);

struct gov_cca_Services__object* skel_pde_PDEdriver_fconnect_gov_cca_Services(
  const char* url, sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct sidl_BaseInterface__object* 
  skel_pde_PDEdriver_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  struct sidl_BaseInterface__object * *_ex);
struct pde_PDEdriver__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
