/*
 * File:          whoc_UIPort_Impl.c
 * Symbol:        whoc.UIPort-v1.0
 * Symbol Type:   class
 * Babel Version: 1.2.0
 * Description:   Server-side implementation for whoc.UIPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.UIPort" (version 1.0)
 */

#include "whoc_UIPort_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.UIPort._includes) */
/* Insert-Code-Here {whoc.UIPort._includes} (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(whoc.UIPort._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._load) */
    /* Insert-Code-Here {whoc.UIPort._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__ctor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._ctor) */
    /* Insert-Code-Here {whoc.UIPort._ctor} (constructor method) */
    /*
     * // boilerplate constructor
     * struct whoc_UIPort__data *dptr = (struct whoc_UIPort__data*)malloc(sizeof(struct whoc_UIPort__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct whoc_UIPort__data));
     *   // initialize elements of dptr here
     * }
     * whoc_UIPort__set_data(self, dptr);
     */

    /* DO-NOT-DELETE splicer.end(whoc.UIPort._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__ctor2(
  /* in */ whoc_UIPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._ctor2) */
    /* Insert-Code-Here {whoc.UIPort._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__dtor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._dtor) */
    /* Insert-Code-Here {whoc.UIPort._dtor} (destructor method) */
    /*
     * // boilerplate destructor
     * struct whoc_UIPort__data *dptr = whoc_UIPort__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   whoc_UIPort__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(whoc.UIPort._dtor) */
  }
}

/*
 * Method:  ui[]
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort_ui"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_whoc_UIPort_ui(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort.ui) */
    /* Insert-Code-Here {whoc.UIPort.ui} (ui method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort.ui) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

