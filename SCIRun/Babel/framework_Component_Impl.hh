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
// File:          framework_Component_Impl.hh
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_framework_Component_Impl_hh
#define included_framework_Component_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
#endif
#ifndef included_framework_Component_IOR_h
#include "framework_Component_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_framework_Component_hh
#include "framework_Component.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.Component._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.Component._includes)

namespace framework { 

  /**
   * Symbol "framework.Component" (version 1.0)
   */
  class Component_impl
  // DO-NOT-DELETE splicer.begin(framework.Component._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.Component._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Component self;

    // DO-NOT-DELETE splicer.begin(framework.Component._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.Component._implementation)

  private:
    // private default constructor (required)
    Component_impl() 
    {} 

  public:
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Component_impl( struct framework_Component__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Component_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // static class initializer
    static void _load();

  public:


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
    setServices (
      /* in */ ::gov::cca::Services services
    )
    throw () 
    ;

  };  // end class Component_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Component._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.Component._misc)

#endif
