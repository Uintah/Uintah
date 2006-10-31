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
// File:          who_Com_Impl.cxx
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for who.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "who_Com_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(who.Com._includes)
#include "who.hxx"
#include <SCIRun/Babel/framework_TypeMap.hxx>
// DO-NOT-DELETE splicer.end(who.Com._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
who::Com_impl::Com_impl() : StubBase(reinterpret_cast< 
  void*>(::who::Com::_wrapObj(this)),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(who.Com._ctor2)
  // Insert-Code-Here {who.Com._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(who.Com._ctor2)
}

// user defined constructor
void who::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(who.Com._ctor)
  // Insert-Code-Here {who.Com._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(who.Com._ctor)
}

// user defined destructor
void who::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(who.Com._dtor)
  // Insert-Code-Here {who.Com._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(who.Com._dtor)
}

// static class initializer
void who::Com_impl::_load() {
  // DO-NOT-DELETE splicer.begin(who.Com._load)
  // Insert-Code-Here {who.Com._load} (class initialization)
  // DO-NOT-DELETE splicer.end(who.Com._load)
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
who::Com_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
{
  // DO-NOT-DELETE splicer.begin(who.Com.setServices)
  svc = services;
  ::who::IDPort idp = ::who::IDPort::_create();
  ::who::UIPort uip = ::who::UIPort::_create();

  ::framework::TypeMap tm = ::framework::TypeMap::_create();
  ::gov::cca::Port idPort = ::sidl::babel_cast< ::gov::cca::Port>(idp);
  if (idPort._not_nil()) {
    svc.addProvidesPort(idPort, "idport", "gov.cca.ports.IDPort", tm);
  }

  UCXX ::gov::cca::Port uiPort = ::sidl::babel_cast< ::gov::cca::Port>(uip);
  if (uiPort._not_nil()) {
    svc.addProvidesPort(uiPort, "ui", "gov.cca.ports.UIPort", tm);
  }
  // DO-NOT-DELETE splicer.end(who.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(who.Com._misc)
// Insert-Code-Here {who.Com._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(who.Com._misc)

