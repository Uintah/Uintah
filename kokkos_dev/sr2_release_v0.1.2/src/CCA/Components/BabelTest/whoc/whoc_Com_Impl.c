/*
 * File:          whoc_Com_Impl.c
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 1.2.0
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.Com" (version 1.0)
 */

#include "whoc_Com_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.Com._includes) */
#include <stdio.h>
#include <stdlib.h>

#include "whoc_IDPort.h"
/* DO-NOT-DELETE splicer.end(whoc.Com._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_Com__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.Com._load) */
    /* Insert-Code-Here {whoc.Com._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.Com._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_Com__ctor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.Com._ctor) */
    struct whoc_Com__data* data = (struct whoc_Com__data*) malloc(sizeof(struct whoc_Com__data));
    data->services = NULL;

    whoc_Com__set_data(self, data);
    /* DO-NOT-DELETE splicer.end(whoc.Com._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_Com__ctor2(
  /* in */ whoc_Com self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.Com._ctor2) */
    /* Insert-Code-Here {whoc.Com._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.Com._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_Com__dtor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.Com._dtor) */
    struct whoc_Com__data* data = whoc_Com__get_data(self);

    if (data->services != NULL) {
      gov_cca_Services_deleteRef(data->services, _ex);
    }

    if (data) {
      free((void*) data);
      whoc_Com__set_data(self, NULL);
    }
    /* DO-NOT-DELETE splicer.end(whoc.Com._dtor) */
  }
}

/*
 *  Starts up a component presence in the calling framework.
 * @param Svc the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com_setServices"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_Com_setServices(
  /* in */ whoc_Com self,
  /* in */ gov_cca_Services services,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.Com.setServices) */
  struct whoc_Com__data* data = whoc_Com__get_data(self);
  sidl_BaseInterface ex;
  gov_cca_Port ip;
  whoc_IDPort idPort = whoc_IDPort__create(&ex);
  ip = gov_cca_Port__cast(idPort, _ex);
  SIDL_CHECK(*_ex);

  gov_cca_TypeMap properties = gov_cca_Services_createTypeMap(services, &ex);
  gov_cca_Services_addProvidesPort(services, ip, "IDPort", "gov.cca.ports.IDPort", properties, &ex);
  gov_cca_Port_deleteRef(ip, _ex);
  SIDL_CHECK(*_ex);

  data->services = services;
  gov_cca_Services_addRef(services, _ex);
  SIDL_CHECK(*_ex);
 EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.Com.setServices) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

