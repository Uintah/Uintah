//
// For more information, please see: http://software.sci.utah.edu
//
// The MIT License
//
// Copyright (c) 2005 Scientific Computing and Imaging Institute,
// University of Utah.
//
// 
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

// File:          hello_IDPort_Impl.cxx
// Symbol:        hello.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for hello.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "hello_IDPort_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(hello.IDPort._includes)
// Insert-Code-Here {hello.IDPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(hello.IDPort._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
hello::IDPort_impl::IDPort_impl() : StubBase(reinterpret_cast< 
  void*>(::hello::IDPort::_wrapObj(this)),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(hello.IDPort._ctor2)
  // Insert-Code-Here {hello.IDPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(hello.IDPort._ctor2)
}

// user defined constructor
void hello::IDPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._ctor)
  // Insert-Code-Here {hello.IDPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(hello.IDPort._ctor)
}

// user defined destructor
void hello::IDPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._dtor)
  // Insert-Code-Here {hello.IDPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(hello.IDPort._dtor)
}

// static class initializer
void hello::IDPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(hello.IDPort._load)
  // Insert-Code-Here {hello.IDPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(hello.IDPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 *  Test prot. Return a string as an ID for Hello component
 */
::std::string
hello::IDPort_impl::getID_impl () 

{
  // DO-NOT-DELETE splicer.begin(hello.IDPort.getID)
  // Insert-Code-Here {hello.IDPort.getID} (getID method)
    return ::std::string("hello::IDPort_impl::getID_Impl()");
  // DO-NOT-DELETE splicer.end(hello.IDPort.getID)
}


// DO-NOT-DELETE splicer.begin(hello.IDPort._misc)
// Insert-Code-Here {hello.IDPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(hello.IDPort._misc)

