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
 * Babel Version: 0.99.2
 * Description:   Server-side implementation for whoc.UIPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_whoc_UIPort_Impl_h
#define included_whoc_UIPort_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_gov_cca_Port_h
#include "gov_cca_Port.h"
#endif
#ifndef included_gov_cca_ports_UIPort_h
#include "gov_cca_ports_UIPort.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_whoc_UIPort_h
#include "whoc_UIPort.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.UIPort._includes) */
/* Insert-Code-Here {whoc.UIPort._includes} (include files) */
/* DO-NOT-DELETE splicer.end(whoc.UIPort._includes) */

/*
 * Private data for class whoc.UIPort
 */

struct whoc_UIPort__data {
  /* DO-NOT-DELETE splicer.begin(whoc.UIPort._data) */
  /* Insert-Code-Here {whoc.UIPort._data} (private data members) */
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

extern
void
impl_whoc_UIPort__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_UIPort__ctor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_UIPort__ctor2(
  /* in */ whoc_UIPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_UIPort__dtor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct gov_cca_Port__object* 
  impl_whoc_UIPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_UIPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fconnect_gov_cca_ports_UIPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fcast_gov_cca_ports_UIPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_UIPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_UIPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_UIPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_UIPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_UIPort__object* impl_whoc_UIPort_fconnect_whoc_UIPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_UIPort__object* impl_whoc_UIPort_fcast_whoc_UIPort(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_whoc_UIPort_ui(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex);

extern struct gov_cca_Port__object* 
  impl_whoc_UIPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_UIPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fconnect_gov_cca_ports_UIPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fcast_gov_cca_ports_UIPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_UIPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_UIPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_UIPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_UIPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_UIPort__object* impl_whoc_UIPort_fconnect_whoc_UIPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_UIPort__object* impl_whoc_UIPort_fcast_whoc_UIPort(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
