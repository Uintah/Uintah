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


#ifndef EP_CHANNEL_INTERFACE_H
#define EP_CHANNEL_INTERFACE_H 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

  class Message;
  class SpChannel;
  class TypeInfo;

  /**************************************
 
  CLASS
  EpChannel
   
  DESCRIPTION
     The base class for all communication-specific endpoint 
     channel abstractions. An endpoint channel is a server
     channel in the sense that waits for one (or more depending
     on implementation) clients to bind to it. It also contains
     handler functions which are invoked when a message bound 
     for them is recieved.

  ****************************************/

  class EpChannel {
  public:

    //Handler function type
    typedef void (*HPF)(Message* ); 

    //Initializing methods used to establish the
    //connection. These are called from the PIDL.
    virtual void openConnection() = 0;
    virtual void activateConnection(void* obj) = 0;
    virtual void closeConnection() = 0;
    virtual std::string getUrl() = 0; 
  
    ///////////////
    // Retrieves a Message associated with this particular object.
    virtual Message* getMessage() = 0;

    //////////////
    // Used in order to directly bind a SpChannel to
    // this EpChannel.
    virtual void bind(SpChannel* spchan) = 0;

    //////////////
    // Functions used to establish the handler functions
    virtual void allocateHandlerTable(int size)=0;
    virtual void registerHandler(int num, void* handle)=0;
  
    //////////
    // A pointer to the type information.
    const TypeInfo* d_typeinfo;
  };
}

#endif
