/*
 * File:          whoc_Com_Impl.h
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

#ifndef included_whoc_Com_Impl_h
#define included_whoc_Com_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_gov_cca_Services_h
#include "gov_cca_Services.h"
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

extern void
impl_whoc_Com__ctor(
  whoc_Com);

extern void
impl_whoc_Com__dtor(
  whoc_Com);

/*
 * User-defined object methods
 */

extern void
impl_whoc_Com_setServices(
  whoc_Com,
  gov_cca_Services);

#ifdef __cplusplus
}
#endif
#endif
