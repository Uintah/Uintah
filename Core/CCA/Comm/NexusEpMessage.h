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


/**************************************
 
CLASS
   NexusEpMessage
   
DESCRIPTION
   A Globus-Nexus implementation of the Message
   abstract interface. This implementation uses
   both a startpoint and endpoint message. 

SEE ALSO
   EpChannel.h
   Message.h

****************************************/

#ifndef NEXUS_EP_MESSAGE_H
#define NEXUS_EP_MESSAGE_H 

#include <Core/CCA/Comm/Message.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>

namespace SCIRun {
  class ReplyEP;
  class NexusEpMessage : public Message {
  public:

    NexusEpMessage(globus_nexus_endpoint_t , globus_nexus_buffer_t msgbuff);
    virtual ~NexusEpMessage();
    void* getLocalObj();

    int getRecvBufferCopy(void* buf);
    int getSendBufferCopy(void* buf);
    void setRecvBuffer(void* buf, int len);
    void setSendBuffer(void* buf, int len);

    void createMessage();
    void marshalInt(const int *i, int size = 1);
    void marshalByte(const char *b, int size = 1);
    void marshalChar(const char *c,int size = 1);
    void marshalFloat(const float *f, int size = 1);
    void marshalDouble(const double *d, int size = 1);
    void marshalLong(const long *l, int size = 1);
    void marshalSpChannel(SpChannel* channel);
    void marshalOpaque(void **ptr, int size=1);

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
    void unmarshalOpaque(void** ptr, int size = 1);
    void destroyMessage();

  private:

    ///////////////////////
    // Buffer upon which we get the starting message
    globus_nexus_buffer_t _buffer;

    //////////////////////
    // Buffer upon which we send the reply on
    globus_nexus_buffer_t* _sendbuff;

    /////////////////////
    // The startpoint to reply to (marshalled to us via _buffer)
    globus_nexus_startpoint_t msg_sp;

    ////////////////////
    // Size of _sendbuff
    int msgsize;

    ///////////////////////
    // The endpoint associated with this Communication Channel
    globus_nexus_endpoint_t d_ep;
  
    /////////////////
    // Reply endpoint to facilitate receiving a reply
    ReplyEP* _reply;

    ////////////////
    // Startpoint which we marshal together with the
    // message in order to recieve a reply.
    globus_nexus_startpoint_t _reply_sp;

    void printDebug( const std::string& );

    //Toggles on/off whether debugging info gets printed
    static const int kDEBUG=0;
  };
}


#endif





