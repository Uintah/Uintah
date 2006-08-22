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

// File:          hello_Com_Impl.cxx
// Symbol:        hello.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for hello.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "hello_Com_Impl.hxx"

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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(hello.Com._includes)
#include "hello.hxx"

#include <SCIRun/Babel/framework_TypeMap.hxx>
// DO-NOT-DELETE splicer.end(hello.Com._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
hello::Com_impl::Com_impl() : StubBase(reinterpret_cast< 
  void*>(::hello::Com::_wrapObj(this)),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(hello.Com._ctor2)
  // Insert-Code-Here {hello.Com._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(hello.Com._ctor2)
}

// user defined constructor
void hello::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.Com._ctor)
  // Insert-Code-Here {hello.Com._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(hello.Com._ctor)
}

// user defined destructor
void hello::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.Com._dtor)
  // Insert-Code-Here {hello.Com._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(hello.Com._dtor)
}

// static class initializer
void hello::Com_impl::_load() {
  // DO-NOT-DELETE splicer.begin(hello.Com._load)
  // Insert-Code-Here {hello.Com._load} (class initialization)
  // DO-NOT-DELETE splicer.end(hello.Com._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 *  Starts up a component presence in the calling framework.
 * @param Svc the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */
void
hello::Com_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
{
  // DO-NOT-DELETE splicer.begin(hello.Com.setServices)
  svc = services;
  ::hello::GoPort gp = ::hello::GoPort::_create();
  gp.setServices(svc);
  ::gov::cca::Port goPort = ::sidl::babel_cast< ::gov::cca::Port>(gp);
  ::framework::TypeMap tm = ::framework::TypeMap::_create();
  if (goPort._not_nil()) {
    svc.addProvidesPort(goPort, std::string("go"), std::string("gov.cca.ports.GoPort"), tm);
  }
  svc.registerUsesPort(std::string("idport"), std::string("gov.cca.ports.IDPort"), tm);
  // DO-NOT-DELETE splicer.end(hello.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(hello.Com._misc)
// Insert-Code-Here {hello.Com._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(hello.Com._misc)

