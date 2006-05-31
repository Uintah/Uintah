/*
 * File:          whoc_IDPort_Impl.h
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.11.0
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.11.0
 */

#ifndef included_whoc_IDPort_Impl_h
#define included_whoc_IDPort_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_gov_cca_Port_h
#include "gov_cca_Port.h"
#endif
#ifndef included_gov_cca_ports_IDPort_h
#include "gov_cca_ports_IDPort.h"
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
#ifndef included_whoc_IDPort_h
#include "whoc_IDPort.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(whoc.IDPort._includes) */

/*
 * Private data for class whoc.IDPort
 */

struct whoc_IDPort__data {
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct whoc_IDPort__data*
whoc_IDPort__get_data(
  whoc_IDPort);

extern void
whoc_IDPort__set_data(
  whoc_IDPort,
  struct whoc_IDPort__data*);

extern
void
impl_whoc_IDPort__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__ctor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__dtor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct gov_cca_Port__object* 
  impl_whoc_IDPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_IDPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fconnect_gov_cca_ports_IDPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fcast_gov_cca_ports_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fconnect_whoc_IDPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fcast_whoc_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern
char*
impl_whoc_IDPort_getID(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

extern struct gov_cca_Port__object* 
  impl_whoc_IDPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_IDPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fconnect_gov_cca_ports_IDPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fcast_gov_cca_ports_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fconnect_whoc_IDPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fcast_whoc_IDPort(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
