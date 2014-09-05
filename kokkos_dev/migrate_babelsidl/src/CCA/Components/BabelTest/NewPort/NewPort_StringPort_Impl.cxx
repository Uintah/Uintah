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
// File:          NewPort_StringPort_Impl.cxx
// Symbol:        NewPort.StringPort-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for NewPort.StringPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "NewPort_StringPort_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(NewPort.StringPort._includes)
// Insert-Code-Here {NewPort.StringPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
NewPort::StringPort_impl::StringPort_impl() : StubBase(reinterpret_cast< 
  void*>(::NewPort::StringPort::_wrapObj(this)),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._ctor2)
  // Insert-Code-Here {NewPort.StringPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._ctor2)
}

// user defined constructor
void NewPort::StringPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._ctor)
  // Insert-Code-Here {NewPort.StringPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._ctor)
}

// user defined destructor
void NewPort::StringPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._dtor)
  // Insert-Code-Here {NewPort.StringPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._dtor)
}

// static class initializer
void NewPort::StringPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._load)
  // Insert-Code-Here {NewPort.StringPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getString[]
 */
::std::string
NewPort::StringPort_impl::getString_impl () 

{
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort.getString)
  // Insert-Code-Here {NewPort.StringPort.getString} (getString method)
  return "NewPort::StringPort_impl::getString CALLED";
  // DO-NOT-DELETE splicer.end(NewPort.StringPort.getString)
}


// DO-NOT-DELETE splicer.begin(NewPort.StringPort._misc)
// Insert-Code-Here {NewPort.StringPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._misc)

