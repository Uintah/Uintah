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
// File:          who_Com_Impl.hh
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

#ifndef included_who_Com_Impl_hh
#define included_who_Com_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_who_Com_IOR_h
#include "who_Com_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif
#ifndef included_who_Com_hh
#include "who_Com.hh"
#endif


// DO-NOT-DELETE splicer.begin(who.Com._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.Com._includes)

namespace who { 

  /**
   * Symbol "who.Com" (version 1.0)
   */
  class Com_impl
  // DO-NOT-DELETE splicer.begin(who.Com._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(who.Com._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Com self;

    // DO-NOT-DELETE splicer.begin(who.Com._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(who.Com._implementation)

  private:
    // private default constructor (required)
    Com_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Com_impl( struct who_Com__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Com_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Obtain Services handle, through which the 
     * component communicates with the framework. 
     * This is the one method that every CCA Component
     * must implement. 
     */
    void
    setServices (
      /*in*/ ::gov::cca::Services services
    )
    throw () 
    ;

  };  // end class Com_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.Com._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(who.Com._misc)

#endif
