/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 * File:          whoc_IDPort_Impl.h
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030915 14:58:56 MST
 * Generated:     20030915 14:59:01 MST
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 7
 * source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
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
