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
 * File:          whoc_Com_Impl.h
 * Symbol:        whoc.Com-v1.0
 * Symbol Type:   class
 * Babel Version: 0.99.2
 * Description:   Server-side implementation for whoc.Com
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_whoc_Com_Impl_h
#define included_whoc_Com_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_gov_cca_Component_h
#include "gov_cca_Component.h"
#endif
#ifndef included_gov_cca_Services_h
#include "gov_cca_Services.h"
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
#ifndef included_whoc_Com_h
#include "whoc_Com.h"
#endif

/* DO-NOT-DELETE splicer.begin(whoc.Com._includes) */
/* Insert-Code-Here {whoc.Com._includes} (include files) */
/* DO-NOT-DELETE splicer.end(whoc.Com._includes) */

/*
 * Private data for class whoc.Com
 */

struct whoc_Com__data {
  /* DO-NOT-DELETE splicer.begin(whoc.Com._data) */
  gov_cca_Services services;
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

extern
void
impl_whoc_Com__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__ctor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__ctor2(
  /* in */ whoc_Com self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_whoc_Com__dtor(
  /* in */ whoc_Com self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct gov_cca_Component__object* 
  impl_whoc_Com_fconnect_gov_cca_Component(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_whoc_Com_fcast_gov_cca_Component(void* bi, sidl_BaseInterface* _ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fconnect_gov_cca_Services(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fcast_gov_cca_Services(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_Com_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_whoc_Com_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_Com_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_whoc_Com_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
extern struct whoc_Com__object* impl_whoc_Com_fconnect_whoc_Com(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_Com__object* impl_whoc_Com_fcast_whoc_Com(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_whoc_Com_setServices(
  /* in */ whoc_Com self,
  /* in */ gov_cca_Services services,
  /* out */ sidl_BaseInterface *_ex);

extern struct gov_cca_Component__object* 
  impl_whoc_Com_fconnect_gov_cca_Component(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Component__object* 
  impl_whoc_Com_fcast_gov_cca_Component(void* bi, sidl_BaseInterface* _ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fconnect_gov_cca_Services(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct gov_cca_Services__object* 
  impl_whoc_Com_fcast_gov_cca_Services(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_whoc_Com_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* impl_whoc_Com_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_whoc_Com_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_whoc_Com_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* impl_whoc_Com_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_whoc_Com_fcast_sidl_RuntimeException(void* bi, sidl_BaseInterface* _ex);
extern struct whoc_Com__object* impl_whoc_Com_fconnect_whoc_Com(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct whoc_Com__object* impl_whoc_Com_fcast_whoc_Com(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
