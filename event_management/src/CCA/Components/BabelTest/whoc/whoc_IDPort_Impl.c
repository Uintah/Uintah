/*
 * File:          whoc_IDPort_Impl.c
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.11.0
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.11.0
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.IDPort" (version 1.0)
 */

#include "whoc_IDPort_Impl.h"

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._includes) */
/* Put additional includes or other arbitrary code here... */
#include <stdio.h>

#include "sidl_String.h"
/* DO-NOT-DELETE splicer.end(whoc.IDPort._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_IDPort__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._load) */
  /* Insert-Code-Here {whoc.IDPort._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_IDPort__ctor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._ctor) */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_IDPort__dtor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._dtor) */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._dtor) */
}

/*
 * Test prot. Return a string as an ID for Hello component
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort_getID"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_whoc_IDPort_getID(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort.getID) */

  return sidl_String_strdup("World (in C)");

  /* DO-NOT-DELETE splicer.end(whoc.IDPort.getID) */
}
/* Babel internal methods, Users should not edit below this line. */
struct gov_cca_Port__object* impl_whoc_IDPort_fconnect_gov_cca_Port(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return gov_cca_Port__connectI(url, ar, _ex);
}
struct gov_cca_Port__object* impl_whoc_IDPort_fcast_gov_cca_Port(void* bi,
  sidl_BaseInterface* _ex) {
  return gov_cca_Port__cast(bi, _ex);
}
struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fconnect_gov_cca_ports_IDPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return gov_cca_ports_IDPort__connectI(url, ar, _ex);
}
struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fcast_gov_cca_ports_IDPort(void* bi,
  sidl_BaseInterface* _ex) {
  return gov_cca_ports_IDPort__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_whoc_IDPort_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_whoc_IDPort_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_whoc_IDPort_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_whoc_IDPort_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct whoc_IDPort__object* impl_whoc_IDPort_fconnect_whoc_IDPort(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return whoc_IDPort__connectI(url, ar, _ex);
}
struct whoc_IDPort__object* impl_whoc_IDPort_fcast_whoc_IDPort(void* bi,
  sidl_BaseInterface* _ex) {
  return whoc_IDPort__cast(bi, _ex);
}
