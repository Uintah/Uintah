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
 * File:          whoc_Com_Impl.h
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20030915 14:58:58 MST
 * Generated:     20030915 14:59:01 MST
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 13
 * source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/whoc/whoc.sidl
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
