/*
 * File:          whoc_IDPort_Impl.h
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 1.2.0
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(whoc.IDPort._hincludes) */

/*
 * Private data for class whoc.IDPort
 */

struct whoc_IDPort__data {
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._data) */
  /* Insert-Code-Here {whoc.IDPort._data} (private data members) */
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
impl_whoc_IDPort__ctor2(
  /* in */ whoc_IDPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__dtor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
char*
impl_whoc_IDPort_getID(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/

/* DO-NOT-DELETE splicer.begin(_hmisc) */
#ifdef __cplusplus
}
#endif
/* DO-NOT-DELETE splicer.end(_hmisc) */

#endif
