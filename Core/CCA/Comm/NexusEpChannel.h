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


#ifndef NEXUS_EP_CHANNEL_H
#define NEXUS_EP_CHANNEL_H 

#include <Core/CCA/Comm/EpChannel.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>

namespace SCIRun {
  class Object;
  /**************************************
 
  CLASS
     NexusEpChannel
   
  DESCRIPTION
     A Globus-Nexus implementation of the EpChannel
     abstract interface. This implementation routs
     all incoming nexus messages to its appropriate
     handler via the unknown_handler. 

  SEE ALSO
     EpChannel.h

  ****************************************/

  class NexusEpChannel : public EpChannel {
  public:

    NexusEpChannel();
    virtual ~NexusEpChannel();
    void openConnection();
    void activateConnection(void *);
    void closeConnection();
    std::string getUrl(); 
    Message* getMessage();
    void allocateHandlerTable(int size);
    void registerHandler(int num, void* handle);
    int approve(globus_nexus_startpoint_t* sp, Object* obj);
    void bind(SpChannel* spchan);

    ////////////
    // The table of handlers from the sidl generated file
    // used to relay a message to them 
    HPF* handler_table;
    int table_size;

    /////////////
    // Buffer in which we store the message we recieve
    globus_nexus_buffer_t msgbuffer;

  private:

    //////////
    // The endpoint associated with this object.
    globus_nexus_endpoint_t d_endpoint;

    void printDebug(const std::string& );
  };
}

#endif
