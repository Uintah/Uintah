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
// File:          framework_ComponentID_Impl.hh
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040129 15:00:03 MST
// Generated:     20040129 15:00:05 MST
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 21
// source-url    = file:/home/sci/damevski/SCIRun/ccadebug-RH8/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_ComponentID_Impl_hh
#define included_framework_ComponentID_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_ComponentID_IOR_h
#include "framework_ComponentID_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_ComponentID_hh
#include "framework_ComponentID.hh"
#endif
#ifndef included_gov_cca_CCAException_hh
#include "gov_cca_CCAException.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.ComponentID._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.ComponentID._includes)

namespace framework { 

  /**
   * Symbol "framework.ComponentID" (version 1.0)
   */
  class ComponentID_impl
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.ComponentID._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    ComponentID self;

    // DO-NOT-DELETE splicer.begin(framework.ComponentID._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.ComponentID._implementation)

  private:
    // private default constructor (required)
    ComponentID_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    ComponentID_impl( struct framework_ComponentID__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~ComponentID_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Returns the instance name provided in 
     * &lt;code&gt;BuilderService.createInstance()&lt;/code&gt;
     * or in 
     * &lt;code&gt;AbstractFramework.getServices()&lt;/code&gt;.
     * @throws CCAException if &lt;code&gt;ComponentID&lt;/code&gt; is invalid
     */
    ::std::string
    getInstanceName() throw ( 
      ::gov::cca::CCAException
    );

    /**
     * Returns a framework specific serialization of the ComponentID.
     * @throws CCAException if &lt;code&gt;ComponentID&lt;/code&gt; is
     * invalid.
     */
    ::std::string
    getSerialization() throw ( 
      ::gov::cca::CCAException
    );
  };  // end class ComponentID_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.ComponentID._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.ComponentID._misc)

#endif
