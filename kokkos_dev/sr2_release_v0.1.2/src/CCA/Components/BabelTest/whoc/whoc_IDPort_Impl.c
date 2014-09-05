/*
 * File:          whoc_IDPort_Impl.c
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 1.2.0
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.IDPort" (version 1.0)
 */

#include "whoc_IDPort_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._includes) */
#include <stdio.h>

#include "sidl_String.h"
/* DO-NOT-DELETE splicer.end(whoc.IDPort._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
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
  {
    /* DO-NOT-DELETE splicer.begin(whoc.IDPort._load) */
    /* Insert-Code-Here {whoc.IDPort._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.IDPort._load) */
  }
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
  {
    /* DO-NOT-DELETE splicer.begin(whoc.IDPort._ctor) */
    /* Insert-Code-Here {whoc.IDPort._ctor} (constructor method) */
    /*
     * // boilerplate constructor
     * struct whoc_IDPort__data *dptr = (struct whoc_IDPort__data*)malloc(sizeof(struct whoc_IDPort__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct whoc_IDPort__data));
     *   // initialize elements of dptr here
     * }
     * whoc_IDPort__set_data(self, dptr);
     */

    /* DO-NOT-DELETE splicer.end(whoc.IDPort._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_IDPort__ctor2(
  /* in */ whoc_IDPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.IDPort._ctor2) */
    /* Insert-Code-Here {whoc.IDPort._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.IDPort._ctor2) */
  }
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
  {
    /* DO-NOT-DELETE splicer.begin(whoc.IDPort._dtor) */
    /* Insert-Code-Here {whoc.IDPort._dtor} (destructor method) */
    /*
     * // boilerplate destructor
     * struct whoc_IDPort__data *dptr = whoc_IDPort__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   whoc_IDPort__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(whoc.IDPort._dtor) */
  }
}

/*
 *  Test prot. Return a string as an ID for Hello component
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
  {
    /* DO-NOT-DELETE splicer.begin(whoc.IDPort.getID) */
    /* Insert-Code-Here {whoc.IDPort.getID} (getID method) */
    return sidl_String_strdup("World (in C)");
    /* DO-NOT-DELETE splicer.end(whoc.IDPort.getID) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

