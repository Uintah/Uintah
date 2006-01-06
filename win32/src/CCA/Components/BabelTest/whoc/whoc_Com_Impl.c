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
 * File:          whoc_Com_Impl.c
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

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.Com" (version 1.0)
 */

#include "whoc_Com_Impl.h"

/* DO-NOT-DELETE splicer.begin(whoc.Com._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(whoc.Com._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__ctor"

void
impl_whoc_Com__ctor(
  whoc_Com self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.Com._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com__dtor"

void
impl_whoc_Com__dtor(
  whoc_Com self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.Com._dtor) */
}

/*
 * Obtain Services handle, through which the 
 * component communicates with the framework. 
 * This is the one method that every CCA Component
 * must implement. 
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_Com_setServices"

void
impl_whoc_Com_setServices(
  whoc_Com self, gov_cca_Services services)
{
  /* DO-NOT-DELETE splicer.begin(whoc.Com.setServices) */

  sidl_BaseException ex;
  gov_cca_TypeMap properties=gov_cca_Services_createTypeMap(
       services, &ex);

  gov_cca_Port idport=gov_cca_Port__cast(whoc_IDPort__create());
  gov_cca_Services_addProvidesPort(services,idport,"idport","gov.cca.ports.IDPort",properties,&ex);
  /* DO-NOT-DELETE splicer.end(whoc.Com.setServices) */
}
