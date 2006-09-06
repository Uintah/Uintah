/*
 * For more information, please see: http://software.sci.utah.edu
 *
 * The MIT License
 *
 * Copyright (c) 2005 Scientific Computing and Imaging Institute,
 * University of Utah.
 *
 * License for the specific language governing rights and limitations under
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * File:          whoc_IDPort_Impl.h
 * Symbol:        whoc.IDPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.99.2
 * Description:   Server-side implementation for whoc.IDPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_whoc_IDPort_Impl_h
#define included_whoc_IDPort_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_gov_cca_Port_h
#include "gov_cca_Port.h"
#endif
#ifndef included_gov_cca_ports_IDPort_h
#include "gov_cca_ports_IDPort.h"
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
#ifndef included_whoc_IDPort_h
#include "whoc_IDPort.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._includes) */
/* Insert-Code-Here {whoc.IDPort._includes} (include files) */
/* DO-NOT-DELETE splicer.end(whoc.IDPort._includes) */

/*
 * Private data for class whoc.IDPort
 */

struct whoc_IDPort__data {
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._data) */
  /* Insert-Code-Here {whoc.IDPort._data} (private data members) */
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

extern
void
impl_whoc_IDPort__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__ctor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__ctor2(
  /* in */ whoc_IDPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_IDPort__dtor(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct gov_cca_Port__object* 
  impl_whoc_IDPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_IDPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fconnect_gov_cca_ports_IDPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fcast_gov_cca_ports_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fconnect_whoc_IDPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fcast_whoc_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern
char*
impl_whoc_IDPort_getID(
  /* in */ whoc_IDPort self,
  /* out */ sidl_BaseInterface *_ex);

extern struct gov_cca_Port__object* 
  impl_whoc_IDPort_fconnect_gov_cca_Port(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Port__object* impl_whoc_IDPort_fcast_gov_cca_Port(void* 
  bi, sidl_BaseInterface* _ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fconnect_gov_cca_ports_IDPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_ports_IDPort__object* 
  impl_whoc_IDPort_fcast_gov_cca_ports_IDPort(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_IDPort_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_IDPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_IDPort_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_IDPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fconnect_whoc_IDPort(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_IDPort__object* impl_whoc_IDPort_fcast_whoc_IDPort(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
