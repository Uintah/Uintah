/*
 * File:          whoc_Com_Impl.h
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 1.2.0
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_whoc_Com_Impl_h
#define included_whoc_Com_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_gov_cca_Component_h
#include "gov_cca_Component.h"
#endif
#ifndef included_gov_cca_Services_h
#include "gov_cca_Services.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_whoc_Com_h
#include "whoc_Com.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.Com._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(whoc.Com._hincludes) */

/*
 * Private data for class whoc.Com
 */

struct whoc_Com__data {
  /* DO-NOT-DELETE splicer.begin(whoc.Com._data) */
  gov_cca_Services services;
  /* DO-NOT-DELETE splicer.end(whoc.Com._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct whoc_Com__data*
whoc_Com__get_data(
  whoc_Com);

extern void
whoc_Com__set_data(
  whoc_Com,
  struct whoc_Com__data*);

extern
void
impl_whoc_Com__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__ctor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__ctor2(
  /* in */ whoc_Com self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__dtor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

#ifdef WITH_RMI
extern struct gov_cca_Services__object* impl_whoc_Com_fconnect_gov_cca_Services(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
void
impl_whoc_Com_setServices(
  /* in */ whoc_Com self,
  /* in */ gov_cca_Services services,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct gov_cca_Services__object* impl_whoc_Com_fconnect_gov_cca_Services(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/

/* DO-NOT-DELETE splicer.begin(_hmisc) */
#ifdef __cplusplus
}
#endif
/* DO-NOT-DELETE splicer.end(_hmisc) */

#endif
