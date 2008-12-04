/*
 * File:          pdeports_PDEdescriptionPort_IOR.h
 * Symbol:        pdeports.PDEdescriptionPort-v0.1
 * Symbol Type:   interface
 * Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
 * Description:   Intermediate Object Representation for pdeports.PDEdescriptionPort
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_pdeports_PDEdescriptionPort_IOR_h
#define included_pdeports_PDEdescriptionPort_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_gov_cca_Port_IOR_h
#include "gov_cca_Port_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "pdeports.PDEdescriptionPort" (version 0.1)
 */

struct pdeports_PDEdescriptionPort__array;
struct pdeports_PDEdescriptionPort__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
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

struct pdeports_PDEdescriptionPort__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ void* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts)(
    /* in */ void* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats)(
    /* in */ void* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in gov.cca.Port-v0.8.1 */
  /* Methods introduced in pdeports.PDEdescriptionPort-v0.1 */
  int32_t (*f_getPDEdescription)(
    /* in */ void* self,
    /* out array<double> */ struct sidl_double__array** nodes,
    /* out array<int> */ struct sidl_int__array** boundaries,
    /* out array<int> */ struct sidl_int__array** dirichletNodes,
    /* out array<double> */ struct sidl_double__array** dirichletValues,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method pre hooks entry point vector.
 */

struct pdeports_PDEdescriptionPort__pre_epv {
  void (*f_getPDEdescription_pre)(
    /* in */ void* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method post hooks entry point vector.
 */

struct pdeports_PDEdescriptionPort__post_epv {
  void (*f_getPDEdescription_post)(
    /* in */ void* self,
    /* in array<double> */ struct sidl_double__array* nodes,
    /* in array<int> */ struct sidl_int__array* boundaries,
    /* in array<int> */ struct sidl_int__array* dirichletNodes,
    /* in array<double> */ struct sidl_double__array* dirichletValues,
    /* in */ int32_t _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Define the interface object structure.
 */

struct pdeports_PDEdescriptionPort__object {
  struct pdeports_PDEdescriptionPort__epv* d_epv;
  void*                                    d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
/*
 * Symbol "pdeports._PDEdescriptionPort" (version 1.0)
 */

struct pdeports__PDEdescriptionPort__array;
struct pdeports__PDEdescriptionPort__object;

/*
 * Declare the method entry point vector.
 */

struct pdeports__PDEdescriptionPort__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__delete)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__exec)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  char* (*f__getURL)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__raddRef)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f__isRemote)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__set_hooks)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__set_contracts)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__dump_stats)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__ctor)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__ctor2)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f__dtor)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in gov.cca.Port-v0.8.1 */
  /* Methods introduced in pdeports.PDEdescriptionPort-v0.1 */
  int32_t (*f_getPDEdescription)(
    /* in */ struct pdeports__PDEdescriptionPort__object* self,
    /* out array<double> */ struct sidl_double__array** nodes,
    /* out array<int> */ struct sidl_int__array** boundaries,
    /* out array<int> */ struct sidl_int__array** dirichletNodes,
    /* out array<double> */ struct sidl_double__array** dirichletValues,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in pdeports._PDEdescriptionPort-v1.0 */
};

/*
 * Define the controls and statistics structure.
 */


struct pdeports__PDEdescriptionPort__cstats {
  sidl_bool use_hooks;
};

/*
 * Define the class object structure.
 */

struct pdeports__PDEdescriptionPort__object {
  struct gov_cca_Port__object                d_gov_cca_port;
  struct pdeports_PDEdescriptionPort__object d_pdeports_pdedescriptionport;
  struct sidl_BaseInterface__object          d_sidl_baseinterface;
  struct pdeports__PDEdescriptionPort__epv*  d_epv;
  void*                                      d_data;
};


struct pdeports__PDEdescriptionPort__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
