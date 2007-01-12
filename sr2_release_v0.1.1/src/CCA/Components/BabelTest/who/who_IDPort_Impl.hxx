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
// File:          who_IDPort_Impl.hxx
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.99.2
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_who_IDPort_Impl_hxx
#define included_who_IDPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_who_IDPort_IOR_h
#include "who_IDPort_IOR.h"
#endif
#ifndef included_gov_cca_ports_IDPort_hxx
#include "gov_cca_ports_IDPort.hxx"
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
#ifndef included_who_IDPort_hxx
#include "who_IDPort.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Insert-Code-Here {who.IDPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

namespace who { 

  /**
   * Symbol "who.IDPort" (version 1.0)
   */
  class IDPort_impl : public virtual ::who::IDPort 
  // DO-NOT-DELETE splicer.begin(who.IDPort._inherits)
  // Insert-Code-Here {who.IDPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.IDPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(who.IDPort._implementation)
    // Insert-Code-Here {who.IDPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(who.IDPort._implementation)

    bool _wrapped;
  public:
    // default constructor, used for data wrapping(required)
    IDPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    IDPort_impl( struct who_IDPort__object * s ) : StubBase(s,true),
      _wrapped(false) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~IDPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     *  Test prot. Return a string as an ID for Hello component
     */
    ::std::string
    getID_impl() ;
  };  // end class IDPort_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.IDPort._misc)
// Insert-Code-Here {who.IDPort._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(who.IDPort._misc)

#endif
