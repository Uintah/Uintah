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

// 
// File:          who_UIPort_Impl.cc
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030915 14:59:08 MST
// Generated:     20030915 14:59:12 MST
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 10
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 
#include "who_UIPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
// Put additional includes or other arbitrary code here...
#include <iostream.h>
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

// user defined constructor
void who::UIPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.UIPort._ctor)
}

// user defined destructor
void who::UIPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.UIPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  ui[]
 */
int32_t
who::UIPort_impl::ui () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(who.UIPort.ui)
  cerr<<" UI button is clicked!"<<endl;
  return 0;	
  // DO-NOT-DELETE splicer.end(who.UIPort.ui)
}


// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

