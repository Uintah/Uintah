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

// File:          framework_ComponentID_Impl.cc
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "framework_ComponentID_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.ComponentID._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.ComponentID._includes)

// user-defined constructor.
void framework::ComponentID_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.ComponentID._ctor)
}

// user-defined destructor.
void framework::ComponentID_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.ComponentID._dtor)
}

// static class initializer.
void framework::ComponentID_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._load)
  // Insert-Code-Here {framework.ComponentID._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Returns the instance name provided in 
 * <code>BuilderService.createInstance()</code>
 * or in 
 * <code>AbstractFramework.getServices()</code>.
 * @throws CCAException if <code>ComponentID</code> is invalid
 */
::std::string
framework::ComponentID_impl::getInstanceName ()
throw ( 
  ::gov::cca::CCAException
)
{
  // DO-NOT-DELETE splicer.begin(framework.ComponentID.getInstanceName)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.ComponentID.getInstanceName)
}

/**
 * Returns a framework specific serialization of the ComponentID.
 * @throws CCAException if <code>ComponentID</code> is
 * invalid.
 */
::std::string
framework::ComponentID_impl::getSerialization ()
throw ( 
  ::gov::cca::CCAException
)
{
  // DO-NOT-DELETE splicer.begin(framework.ComponentID.getSerialization)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.ComponentID.getSerialization)
}


// DO-NOT-DELETE splicer.begin(framework.ComponentID._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.ComponentID._misc)

