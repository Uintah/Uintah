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
// File:          framework_Component_Impl.cc
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040129 15:00:03 MST
// Generated:     20040129 15:00:06 MST
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 11
// source-url    = file:/home/sci/damevski/SCIRun/ccadebug-RH8/../src/SCIRun/Babel/framework.sidl
// 
#include "framework_Component_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.Component._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.Component._includes)

// user defined constructor
void framework::Component_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.Component._ctor)
}

// user defined destructor
void framework::Component_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.Component._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Obtain Services handle, through which the 
 * component communicates with the framework. 
 * This is the one method that every CCA Component
 * must implement. 
 */
void
framework::Component_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Component.setServices)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.Component.setServices)
}


// DO-NOT-DELETE splicer.begin(framework.Component._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.Component._misc)

