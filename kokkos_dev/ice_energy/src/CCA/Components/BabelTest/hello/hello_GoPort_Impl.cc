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
// File:          hello_GoPort_Impl.cc
// Symbol:        hello.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040303 14:22:57 MST
// Generated:     20040303 14:23:03 MST
// Description:   Server-side implementation for hello.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/hello/hello.sidl
// 
#include "hello_GoPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(hello.GoPort._includes)
#include "gov_cca_ports_GoPort.hh"
#include "gov_cca_ports_IDPort.hh"
#include <iostream.h>
// DO-NOT-DELETE splicer.end(hello.GoPort._includes)

// user defined constructor
void hello::GoPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(hello.GoPort._ctor)
}

// user defined destructor
void hello::GoPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.GoPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(hello.GoPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setService[]
 */
void
hello::GoPort_impl::setService (
  /*in*/ ::gov::cca::Services svc ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.setService)
  this->svc=svc;
  // DO-NOT-DELETE splicer.end(hello.GoPort.setService)
}

/**
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
hello::GoPort_impl::go () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(hello.GoPort.go)
  cerr<<"hello::GoPort::go() starts..."<<endl;
  gov::cca::ports::IDPort s=svc.getPort("idport");
  if(!s._is_nil()) cerr<<"Hello "<<s.getID()<<endl;
  else cerr<<"getPort() returns null"<<endl;
  // DO-NOT-DELETE splicer.end(hello.GoPort.go)
}


// DO-NOT-DELETE splicer.begin(hello.GoPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(hello.GoPort._misc)

