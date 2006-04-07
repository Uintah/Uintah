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
// File:          hello_GoPort_Impl.cxx
// Symbol:        hello.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for hello.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "hello_GoPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
// DO-NOT-DELETE splicer.begin(hello.GoPort._includes)
//#include "gov_cca_ports_GoPort.hh"
#include <iostream.h>

#include "sidl_String.h"
#include "gov_cca_ports_IDPort.hxx"
// DO-NOT-DELETE splicer.end(hello.GoPort._includes)

// user defined constructor
void hello::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._ctor)
  // Insert-Code-Here {hello.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(hello.GoPort._ctor)
}

// user defined destructor
void hello::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._dtor)
  // Insert-Code-Here {hello.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(hello.GoPort._dtor)
}

// static class initializer
void hello::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._load)
  // Insert-Code-Here {hello.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(hello.GoPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setServices[]
 */
void
hello::GoPort_impl::setServices_impl (
  /* in */UCXX ::gov::cca::Services services ) 
{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.setServices)
  this->svc = services;
  // DO-NOT-DELETE splicer.end(hello.GoPort.setServices)
}

/**
 * Execute some encapsulated functionality on the component.
 * Return 0 if ok, -1 if internal error but component may be
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
hello::GoPort_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.go)
  ::std::cerr << "hello::GoPort::go() starts..." << ::std::endl;
  UCXX ::gov::cca::Port p = svc.getPort("idport");
  UCXX ::gov::cca::ports::IDPort s = UCXX ::sidl::babel_cast<UCXX ::gov::cca::ports::IDPort>(p);
  if (s._not_nil()) {
    ::std::cerr << "Hello " << s.getID() << ::std::endl;
  } else {
    ::std::cerr << "getPort() returns null" << ::std::endl;
  }
  return 0;
  // DO-NOT-DELETE splicer.end(hello.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(hello.GoPort._misc)
// Insert-Code-Here {hello.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(hello.GoPort._misc)

