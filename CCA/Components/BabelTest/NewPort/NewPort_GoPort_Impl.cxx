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
// File:          NewPort_GoPort_Impl.cc
// Symbol:        NewPort.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for NewPort.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "NewPort_GoPort_Impl.hxx"

// DO-NOT-DELETE splicer.begin(NewPort.GoPort._includes)
#include "NewPort.hxx"
#include <iostream>
// DO-NOT-DELETE splicer.end(NewPort.GoPort._includes)

// user-defined constructor.
void NewPort::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._ctor)
  // Insert-Code-Here {NewPort.GoPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._ctor)
}

// user-defined destructor.
void NewPort::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._dtor)
  // Insert-Code-Here {NewPort.GoPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._dtor)
}

// static class initializer.
void NewPort::GoPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort._load)
  // Insert-Code-Here {NewPort.GoPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.GoPort._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setServices[]
 */
void
NewPort::GoPort_impl::setServices (
  /* in */ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.setServices)
  this->svc = svc;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.setServices)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
NewPort::GoPort_impl::go ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(NewPort.GoPort.go)
  UCXX ::gov::cca::Port p = svc.getPort("ustrport");
  UCXX ::NewPort::StringPort s = UCXX ::sidl::babel_cast<UCXX ::NewPort::StringPort>(p);
  std::cerr << "Got the port\n";
  if (!s._is_nil()) {
    std::cerr<<"Received "<< s.getString() <<"\n";
  } else {
    std::cerr<<"getPort() returns null\n";
  }
  return 0;
  // DO-NOT-DELETE splicer.end(NewPort.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(NewPort.GoPort._misc)
// Insert-Code-Here {NewPort.GoPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.GoPort._misc)

