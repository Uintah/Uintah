/*
 * File:          whoc_UIPort_Impl.h
 * Symbol:        whoc.UIPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030227 09:45:28 MST
 * Generated:     20030227 09:45:31 MST
 * Description:   Server-side implementation for whoc.UIPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 10
 * source-url    = file:/home/kzhang/SCIRun/src/Babel/Components/whoc/whoc.sidl
 */

#ifndef included_whoc_UIPort_Impl_h
#define included_whoc_UIPort_Impl_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_whoc_UIPort_h
#include "whoc_UIPort.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.UIPort._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(whoc.UIPort._includes) */

/*
 * Private data for class whoc.UIPort
 */

struct whoc_UIPort__data {
  /* DO-NOT-DELETE splicer.begin(whoc.UIPort._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(whoc.UIPort._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct whoc_UIPort__data*
whoc_UIPort__get_data(
  whoc_UIPort);

extern void
whoc_UIPort__set_data(
  whoc_UIPort,
  struct whoc_UIPort__data*);

extern void
impl_whoc_UIPort__ctor(
  whoc_UIPort);

extern void
impl_whoc_UIPort__dtor(
  whoc_UIPort);

/*
 * User-defined object methods
 */

extern int32_t
impl_whoc_UIPort_ui(
  whoc_UIPort);

#ifdef __cplusplus
}
#endif
#endif
