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
// File:          who_Com_Impl.cc
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030915 14:59:09 MST
// Generated:     20030915 14:59:12 MST
// Description:   Server-side implementation for who.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 13
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 
#include "who_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(who.Com._includes)
#include "who.hh"
// DO-NOT-DELETE splicer.end(who.Com._includes)

// user defined constructor
void who::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.Com._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(who.Com._ctor)
}

// user defined destructor
void who::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.Com._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(who.Com._dtor)
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
who::Com_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(who.Com.setServices)
  who::IDPort idPort=who::IDPort::_create();
  services.addProvidesPort(idPort,"idport","gov.cca.ports.IDPort",0);

  who::UIPort uiPort=who::UIPort::_create();
  services.addProvidesPort(uiPort,"ui","gov.cca.ports.UIPort",0);
  // DO-NOT-DELETE splicer.end(who.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(who.Com._misc)
// DO-NOT-DELETE splicer.end(who.Com._misc)

