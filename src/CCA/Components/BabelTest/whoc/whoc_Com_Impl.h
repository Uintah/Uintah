/*
 * File:          whoc_Com_Impl.h
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 0.11.0
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.11.0
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

/* DO-NOT-DELETE splicer.begin(whoc.Com._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(whoc.Com._includes) */

/*
 * Private data for class whoc.Com
 */

struct whoc_Com__data {
  /* DO-NOT-DELETE splicer.begin(whoc.Com._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
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
impl_whoc_Com__dtor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct gov_cca_Component__object* 
  impl_whoc_Com_fconnect_gov_cca_Component(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_whoc_Com_fcast_gov_cca_Component(void* bi, sidl_BaseInterface* _ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fconnect_gov_cca_Services(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fcast_gov_cca_Services(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_Com_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_whoc_Com_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_Com_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_whoc_Com_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
extern struct whoc_Com__object* impl_whoc_Com_fconnect_whoc_Com(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_Com__object* impl_whoc_Com_fcast_whoc_Com(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_whoc_Com_setServices(
  /* in */ whoc_Com self,
  /* in */ gov_cca_Services services,
  /* out */ sidl_BaseInterface *_ex);

extern struct gov_cca_Component__object* 
  impl_whoc_Com_fconnect_gov_cca_Component(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_whoc_Com_fcast_gov_cca_Component(void* bi, sidl_BaseInterface* _ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fconnect_gov_cca_Services(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fcast_gov_cca_Services(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_Com_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_whoc_Com_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_Com_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_whoc_Com_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
extern struct whoc_Com__object* impl_whoc_Com_fconnect_whoc_Com(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_Com__object* impl_whoc_Com_fcast_whoc_Com(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
