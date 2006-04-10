/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  ServerContext.h: Local class for PIDL that holds the context
 *                   for server objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_ServerContext_h
#define CCA_PIDL_ServerContext_h

#include <Core/CCA/PIDL/Object.h>
#include <Core/CCA/Comm/EpChannel.h>
#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/PIDL/HandlerStorage.h>
//#include <Core/CCA/PIDL/HandlerGateKeeper.h>

namespace SCIRun {
/**************************************
 
CLASS
   ServerContext
   
KEYWORDS
   PIDL, Server
   
DESCRIPTION
   One of these objects is associated with each server object.  It provides
   the state necessary for the PIDL internals, including the comm channel
   associated with this object, a pointer to type information, the objects
   id from the object wharehouse, a pointer to the base object class
   (Object), and to the most-derived type (a void*).  
****************************************/
  struct ServerContext {

    
    //////////
    // A pointer to the type information.
    const TypeInfo* d_typeinfo;

    //////////
    // The Comm Channel associated with this object
    EpChannel* chan;

    //////////
    // The ID of this object from the object wharehouse.  This
    // id is unique within this process.
    int d_objid;

    //////////
    // A pointer to the object base class.
    Object* d_objptr;

    //////////
    // A pointer to the most derived type.  This is used only by
    // sidl generated code.
    void* d_ptr;

    //////////
    // A flag, true if the endpoint has been created for this
    // object.
    bool d_endpoint_active;

    //////////
    // Create the endpoint for this object.
    void activateEndpoint();

    //////////
    // Bind this reference's startpoint to my endpoint
    void bind(Reference& ref);

    //////////
    // A MxN Distribution Scheduler in case this object needs
    // to redistribute an array
    MxNScheduler* d_sched;

    /////////
    // Storage for buffer data between two different handler invocations
    HandlerStorage* storage;

    /////////
    // Storage for buffer data between two different handler invocations
    //    HandlerGateKeeper* gatekeeper;
    
  };
} // End namespace SCIRun

#endif




