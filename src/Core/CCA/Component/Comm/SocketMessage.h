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

#ifndef SOCKET_MESSAGE_H
#define SOCKET_MESSAGE_H 

#include <stdlib.h>

///////////////////////////////
//message layout
//   [size(long)] [id (int)] [other marshaled segements]
//
//

#include <Core/CCA/Component/Comm/Message.h>

namespace SCIRun {
  class SocketEpChannel;
  class SocketSpChannel;
  class SocketMessage : public Message {
  public:
    SocketMessage(int sockfd, void *msg=NULL);

    virtual ~SocketMessage();
    void* getLocalObj();
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


    void setSocketEp(SocketEpChannel* ep);
    void setSocketSp(SocketSpChannel* sp);

    inline static int sendall(int sockfd, void *buf, int *len);

  private:
    inline void marshalBuf(const void *buf, int fullsize);
    inline void unmarshalBuf(void *buf, int fullsize);
    void *msg;
    SocketEpChannel *ep;
    SocketSpChannel *sp;
    int capacity;
    int msg_size;
    static const int INIT_SIZE=1024;
    int sockfd;  //the socket file descreptor through which the message is transmitted.
  };
}

#endif
