/*
 * File:          whoc_Com_Impl.c
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030618 13:12:27 MDT
 * Generated:     20030618 13:12:32 MDT
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 13
 * source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.Com" (version 1.0)
 */

#include "whoc_Com_Impl.h"

/* DO-NOT-DELETE splicer.begin(whoc.Com._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(whoc.Com._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__ctor"

void
impl_whoc_Com__ctor(
  whoc_Com self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.Com._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__dtor"

void
impl_whoc_Com__dtor(
  whoc_Com self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.Com._dtor) */
}

/*
 * Obtain Services handle, through which the 
 * component communicates with the framework. 
 * This is the one method that every CCA Component
 * must implement. 
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com_setServices"

void
impl_whoc_Com_setServices(
  whoc_Com self, gov_cca_Services services)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com.setServices) */

  SIDL_BaseException ex;
  gov_cca_TypeMap properties=gov_cca_Services_createTypeMap(
       services, &ex);

  gov_cca_Port idport=gov_cca_Port__cast(whoc_IDPort__create());
  gov_cca_Services_addProvidesPort(services,idport,"idport","gov.cca.ports.IDPort",properties,&ex);
  /* DO-NOT-DELETE splicer.end(whoc.Com.setServices) */
}
