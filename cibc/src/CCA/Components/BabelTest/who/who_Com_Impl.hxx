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
// File:          who_Com_Impl.hxx
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for who.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_who_Com_Impl_hxx
#define included_who_Com_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_who_Com_IOR_h
#include "who_Com_IOR.h"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
#ifndef included_who_Com_hxx
#include "who_Com.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.Com._includes)
// Insert-Code-Here {who.Com._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.Com._includes)

namespace who { 

  /**
   * Symbol "who.Com" (version 1.0)
   */
  class Com_impl : public virtual ::who::Com 
  // DO-NOT-DELETE splicer.begin(who.Com._inherits)
  // Insert-Code-Here {who.Com._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.Com._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(who.Com._implementation)
    ::gov::cca::Services svc;
    // DO-NOT-DELETE splicer.end(who.Com._implementation)

    bool _wrapped;
  public:
    // default constructor, used for data wrapping(required)
    Com_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Com_impl( struct who_Com__object * s ) : StubBase(s,true),
      _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Com_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


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
    setServices_impl (
      /* in */::gov::cca::Services services
    )
    ;

  };  // end class Com_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.Com._misc)
// Insert-Code-Here {who.Com._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(who.Com._misc)

#endif
