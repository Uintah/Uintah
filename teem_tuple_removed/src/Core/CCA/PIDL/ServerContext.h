/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Core/CCA/PIDL/HandlerGateKeeper.h>

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
    HandlerGateKeeper* gatekeeper;
    
  };
} // End namespace SCIRun

#endif




