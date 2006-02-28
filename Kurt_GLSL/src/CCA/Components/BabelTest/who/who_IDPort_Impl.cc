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
// File:          who_IDPort_Impl.cc
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030915 14:59:02 MST
// Generated:     20030915 14:59:12 MST
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 
#include "who_IDPort_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

// user defined constructor
void who::IDPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.IDPort._ctor)
}

// user defined destructor
void who::IDPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.IDPort._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.IDPort._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Test prot. Return a string as an ID for Hello component
 */
::std::string
who::IDPort_impl::getID () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(who.IDPort.getID)
  return "World (in C++)";
  // DO-NOT-DELETE splicer.end(who.IDPort.getID)
}


// DO-NOT-DELETE splicer.begin(who.IDPort._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(who.IDPort._misc)

