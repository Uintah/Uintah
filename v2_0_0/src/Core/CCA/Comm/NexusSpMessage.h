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

#ifndef NEXUS_SP_MESSAGE_H
#define NEXUS_SP_MESSAGE_H 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

/**************************************
 
CLASS
   NexusSpMessage
   
DESCRIPTION
   A Globus-Nexus implementation of the Message
   abstract interface. This implementation uses
   both a startpoint and endpoint message. 

SEE ALSO
   SpChannel.h
   Message.h

****************************************/

#include <Core/CCA/Comm/Message.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>

namespace SCIRun {
  class ReplyEP;
  class NexusSpMessage : public Message {
  public:

    NexusSpMessage(globus_nexus_startpoint_t* );
    virtual ~NexusSpMessage();
    
    //////////////////
    // The SP does not have a local object so lets try to
    // get the object from the EP that this SP is associated.
    void* getLocalObj();

    int getRecvBufferCopy(void* buf);
    int getSendBufferCopy(void* buf);
    void setRecvBuffer(void* buf, int len);
    void setSendBuffer(void* buf, int len);

    void createMessage();
    void marshalInt(const int *i, int size = 1);
    void marshalSpChannel(SpChannel* channel);
    void marshalByte(const char *b, int size = 1);
    void marshalChar(const char *c, int size = 1);
    void marshalFloat(const float *f, int size = 1);
    void marshalDouble(const double *d, int size = 1);
    void marshalLong(const long *l, int size = 1);
    void sendMessage(int handler);
    void waitReply();
    void unmarshalReply();
    void unmarshalInt(int *i, int size = 1); 
    void unmarshalByte(char *b, int size = 1);
    void unmarshalChar(char *c, int size = 1);
    void unmarshalFloat(float *f, int size = 1);
    void unmarshalDouble(double *d, int size = 1);
    void unmarshalLong(long *l, int size = 1);
    void unmarshalSpChannel(SpChannel* channel);
    void destroyMessage();


    ////////////////
    // Buffer for original message
    globus_nexus_buffer_t* _buffer;

    ///////////////
    // Size of the current message
    int msgsize;

  private:

    /////////////////
    // Reply endpoint to facilitate receiving a reply
    ReplyEP* _reply;

    ////////////////
    // Beffer in which to store the reply
    globus_nexus_buffer_t* _recvbuff;

    ////////////////
    // Startpoint which we marshal together with the
    // message in order to recieve a reply.
    globus_nexus_startpoint_t _reply_sp;

    ///////////////
    // Original startpoint associated with this
    // Communication Channel
    globus_nexus_startpoint_t* d_sp;
  
    void printDebug( const std::string& );
    
    //Toggles on/off whether debugging info gets printed
    static const int kDEBUG=0;
  };
}

#endif
