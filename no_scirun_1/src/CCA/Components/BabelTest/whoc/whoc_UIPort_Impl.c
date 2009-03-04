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
 * File:          whoc_UIPort_Impl.c
 * Symbol:        whoc.UIPort-v1.0
 * Symbol Type:   class
 * Babel Version: 0.99.2
 * Description:   Server-side implementation for whoc.UIPort
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.UIPort" (version 1.0)
 */

#include "whoc_UIPort_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(whoc.UIPort._includes) */
/* Insert-Code-Here {whoc.UIPort._includes} (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(whoc.UIPort._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._load) */
    /* Insert-Code-Here {whoc.UIPort._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__ctor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._ctor) */
    /* Insert-Code-Here {whoc.UIPort._ctor} (constructor method) */
    /*
     * // boilerplate constructor
     * struct whoc_UIPort__data *dptr = (struct whoc_UIPort__data*)malloc(sizeof(struct whoc_UIPort__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct whoc_UIPort__data));
     *   // initialize elements of dptr here
     * }
     * whoc_UIPort__set_data(self, dptr);
     */

    /* DO-NOT-DELETE splicer.end(whoc.UIPort._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__ctor2(
  /* in */ whoc_UIPort self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._ctor2) */
    /* Insert-Code-Here {whoc.UIPort._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_whoc_UIPort__dtor(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort._dtor) */
    /* Insert-Code-Here {whoc.UIPort._dtor} (destructor method) */
    /*
     * // boilerplate destructor
     * struct whoc_UIPort__data *dptr = whoc_UIPort__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   whoc_UIPort__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(whoc.UIPort._dtor) */
  }
}

/*
 * Method:  ui[]
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_UIPort_ui"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_whoc_UIPort_ui(
  /* in */ whoc_UIPort self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(whoc.UIPort.ui) */
    /* Insert-Code-Here {whoc.UIPort.ui} (ui method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(whoc.UIPort.ui) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct gov_cca_Port__object* impl_whoc_UIPort_fconnect_gov_cca_Port(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return gov_cca_Port__connectI(url, ar, _ex);
}
struct gov_cca_Port__object* impl_whoc_UIPort_fcast_gov_cca_Port(void* bi,
  sidl_BaseInterface* _ex) {
  return gov_cca_Port__cast(bi, _ex);
}
struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fconnect_gov_cca_ports_UIPort(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return gov_cca_ports_UIPort__connectI(url, ar, _ex);
}
struct gov_cca_ports_UIPort__object* 
  impl_whoc_UIPort_fcast_gov_cca_ports_UIPort(void* bi,
  sidl_BaseInterface* _ex) {
  return gov_cca_ports_UIPort__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_whoc_UIPort_fconnect_sidl_BaseClass(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_whoc_UIPort_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fconnect_sidl_BaseInterface(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_whoc_UIPort_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_whoc_UIPort_fconnect_sidl_ClassInfo(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_whoc_UIPort_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fconnect_sidl_RuntimeException(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_whoc_UIPort_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct whoc_UIPort__object* impl_whoc_UIPort_fconnect_whoc_UIPort(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return whoc_UIPort__connectI(url, ar, _ex);
}
struct whoc_UIPort__object* impl_whoc_UIPort_fcast_whoc_UIPort(void* bi,
  sidl_BaseInterface* _ex) {
  return whoc_UIPort__cast(bi, _ex);
}
