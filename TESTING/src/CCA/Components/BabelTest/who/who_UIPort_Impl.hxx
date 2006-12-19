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
// File:          who_UIPort_Impl.hxx
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_who_UIPort_Impl_hxx
#define included_who_UIPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_who_UIPort_IOR_h
#include "who_UIPort_IOR.h"
#endif
#ifndef included_gov_cca_ports_UIPort_hxx
#include "gov_cca_ports_UIPort.hxx"
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
#ifndef included_who_UIPort_hxx
#include "who_UIPort.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
// Insert-Code-Here {who.UIPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

namespace who { 

  /**
   * Symbol "who.UIPort" (version 1.0)
   */
  class UIPort_impl : public virtual ::who::UIPort 
  // DO-NOT-DELETE splicer.begin(who.UIPort._inherits)
  // Insert-Code-Here {who.UIPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.UIPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(who.UIPort._implementation)
    // Insert-Code-Here {who.UIPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(who.UIPort._implementation)

    bool _wrapped;
  public:
    // default constructor, used for data wrapping(required)
    UIPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    UIPort_impl( struct who_UIPort__object * s ) : StubBase(s,true),
      _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~UIPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    int32_t
    ui_impl() ;
  };  // end class UIPort_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Insert-Code-Here {who.UIPort._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

#endif
