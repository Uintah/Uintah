//
//  For more information, please see: http://software.sci.utah.edu
// 
//  The MIT License
// 
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
// 
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
// 
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
// 
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
// 

// 
// File:          framework_Component_Impl.cc
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "framework_Component_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.Component._includes)
#include <iostream>
// DO-NOT-DELETE splicer.end(framework.Component._includes)

// user-defined constructor.
void framework::Component_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.Component._ctor)
}

// user-defined destructor.
void framework::Component_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Component._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.Component._dtor)
}

// static class initializer.
void framework::Component_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.Component._load)
  // Insert-Code-Here {framework.Component._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.Component._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Starts up a component presence in the calling framework.
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
framework::Component_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Component.setServices)
  std::cerr << "framework::Component_impl::setServices (does nothing)" << std::endl;
  // DO-NOT-DELETE splicer.end(framework.Component.setServices)
}


// DO-NOT-DELETE splicer.begin(framework.Component._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.Component._misc)

