//
// For more information, please see: http://software.sci.utah.edu
//
// The MIT License
//
// Copyright (c) 2005 Scientific Computing and Imaging Institute,
// University of Utah.
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// 
// File:          who_UIPort_Impl.cxx
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "who_UIPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
#include <iostream>
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
who::UIPort_impl::UIPort_impl() : StubBase(reinterpret_cast< 
  void*>(::who::UIPort::_wrapObj(this)),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(who.UIPort._ctor2)
  // Insert-Code-Here {who.UIPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(who.UIPort._ctor2)
}

// user defined constructor
void who::UIPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._ctor)
  // Insert-Code-Here {who.UIPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(who.UIPort._ctor)
}

// user defined destructor
void who::UIPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._dtor)
  // Insert-Code-Here {who.UIPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(who.UIPort._dtor)
}

// static class initializer
void who::UIPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(who.UIPort._load)
  // Insert-Code-Here {who.UIPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(who.UIPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  ui[]
 */
int32_t
who::UIPort_impl::ui_impl () 

{
  // DO-NOT-DELETE splicer.begin(who.UIPort.ui)
  // Insert-Code-Here {who.UIPort.ui} (ui method)
  std::cerr << " UI button is clicked!" << std::endl;
  return 0;
  // DO-NOT-DELETE splicer.end(who.UIPort.ui)
}


// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Insert-Code-Here {who.UIPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

