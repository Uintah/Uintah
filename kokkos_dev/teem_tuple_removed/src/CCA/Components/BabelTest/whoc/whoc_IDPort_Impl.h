/*
 * File:          whoc_IDPort_Impl.h
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030618 13:12:25 MDT
 * Generated:     20030618 13:12:32 MDT
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 7
 * source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
 */

#ifndef included_whoc_IDPort_Impl_h
#define included_whoc_IDPort_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
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

extern void
impl_whoc_IDPort__ctor(
  whoc_IDPort);

extern void
impl_whoc_IDPort__dtor(
  whoc_IDPort);

/*
 * User-defined object methods
 */

extern char*
impl_whoc_IDPort_getID(
  whoc_IDPort);

#ifdef __cplusplus
}
#endif
#endif
