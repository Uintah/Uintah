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
 * File:          whoc_IDPort_Impl.c
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

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "whoc.IDPort" (version 1.0)
 */

#include "whoc_IDPort_Impl.h"

/* DO-NOT-DELETE splicer.begin(whoc.IDPort._includes) */
/* Put additional includes or other arbitrary code here... */
#include "string.h"
/* DO-NOT-DELETE splicer.end(whoc.IDPort._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__ctor"

void
impl_whoc_IDPort__ctor(
  whoc_IDPort self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort__dtor"

void
impl_whoc_IDPort__dtor(
  whoc_IDPort self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(whoc.IDPort._dtor) */
}

/*
 * Test prot. Return a string as an ID for Hello component
 */

#undef __FUNC__
#define __FUNC__ "impl_whoc_IDPort_getID"

char*
impl_whoc_IDPort_getID(
  whoc_IDPort self)
{
  /* DO-NOT-DELETE splicer.begin(whoc.IDPort.getID) */
  char *id="World (in C )";
  char *s=(char*)malloc(strlen(id)+1);
  strcpy(s, id);
  return s;
  /* DO-NOT-DELETE splicer.end(whoc.IDPort.getID) */
}
