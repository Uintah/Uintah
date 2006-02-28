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
 * File:          whoc_UIPort_Impl.h
 * Symbol:        whoc.UIPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030915 14:58:57 MST
 * Generated:     20030915 14:59:01 MST
 * Description:   Server-side implementation for whoc.UIPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 10
 * source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
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
