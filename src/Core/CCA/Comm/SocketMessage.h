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
 *  SocketMessage.h: Socket implemenation of Message
 *
 *  Written by:
 *   Kosta Damevski and Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMM_SOCKETMESSAGE_H
#define CORE_CCA_COMM_SOCKETMESSAGE_H

#include <Core/CCA/Comm/Message.h>
#include <string>

namespace SCIRun {
  class SocketEpChannel;
  class SocketSpChannel;
  class DTMessage;
  class SocketMessage : public Message {
  public:
    SocketMessage(SocketSpChannel *spchan);
    SocketMessage(DTMessage *dtmsg);
    virtual ~SocketMessage();
    void* getLocalObj();

    int getRecvBufferCopy(void* buf);
    int getSendBufferCopy(void* buf);
    void setRecvBuffer(void* buf, int len);
    void setSendBuffer(void* buf, int len);

    void createMessage();
    void marshalInt(const int *i, int size = 1);
    void marshalByte(const char *b, int size = 1); 
    void marshalChar(const char *c, int size = 1);
    void marshalFloat(const float *f, int size = 1);
    void marshalDouble(const double *d, int size = 1);
    void marshalLong(const long *l, int size = 1);
    void marshalSpChannel(SpChannel* channel);
    void sendMessage(int handler);
    void waitReply();
    void unmarshalReply();
    void unmarshalInt(int *i, int size = 1);
    void unmarshalByte(char* b, int size = 1);
    void unmarshalChar(char *c, int size = 1);
    void unmarshalFloat(float *f, int size = 1);
    void unmarshalDouble(double *d, int size = 1);
    void unmarshalLong(long *l, int size = 1);
    void unmarshalSpChannel(SpChannel* channel);
    void destroyMessage();

  private:
    inline void marshalBuf(const void *buf, int fullsize);
    inline void unmarshalBuf(void *buf, int fullsize);

    ///////////////////////////////
    //message layout
    //  [id (int)] [other marshaled segements]
    void *msg;
    int capacity;
    int msg_size;
    int msg_length;
    static const int INIT_SIZE=1024;
    SocketSpChannel *spchan;
    DTMessage *dtmsg;
  };
}

#endif
