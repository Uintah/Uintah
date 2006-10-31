//
//  For more information, please see: http://software.sci.utah.edu
//
//  The MIT License
//
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//
//  
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


// File:          framework_ComponentID_Impl.hxx
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_framework_ComponentID_Impl_hxx
#define included_framework_ComponentID_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_framework_ComponentID_IOR_h
#include "framework_ComponentID_IOR.h"
#endif
#ifndef included_framework_ComponentID_hxx
#include "framework_ComponentID.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif


// DO-NOT-DELETE splicer.begin(framework.ComponentID._includes)
// Insert-Code-Here {framework.ComponentID._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.ComponentID._includes)

namespace framework { 

  /**
   * Symbol "framework.ComponentID" (version 1.0)
   */
  class ComponentID_impl : public virtual UCXX ::framework::ComponentID 
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._inherits)
  // Insert-Code-Here {framework.ComponentID._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(framework.ComponentID._implementation)
    // Insert-Code-Here {framework.ComponentID._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.ComponentID._implementation)

  public:
    // default constructor, shouldn't be used (required)
    ComponentID_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      ComponentID_impl( struct framework_ComponentID__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~ComponentID_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:


      /**
       * Returns the instance name provided in
       * <code>BuilderService.createInstance()</code>
       * or in
       * <code>AbstractFramework.getServices()</code>.
       * @throws CCAException if <code>ComponentID</code> is invalid
       */
      ::std::string
      getInstanceName_impl() throw ( 
        UCXX ::gov::cca::CCAException, 
        UCXX ::sidl::RuntimeException
      );

      /**
       * Returns a framework specific serialization of the ComponentID.
       * @throws CCAException if <code>ComponentID</code> is
       * invalid.
       */
      ::std::string
      getSerialization_impl() throw ( 
        UCXX ::gov::cca::CCAException, 
        UCXX ::sidl::RuntimeException
      );
    };  // end class ComponentID_impl

  } // end namespace framework

  // DO-NOT-DELETE splicer.begin(framework.ComponentID._misc)
  // Insert-Code-Here {framework.ComponentID._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._misc)

  #endif
