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

#include <Core/CCA/Component/Comm/Message.h>

namespace SCIRun {
  class SocketMessage : public Message {
  public:

    SocketMessage();
    virtual ~SocketMessage();
    void* getLocalObj();
    void createMessage();
    void marshalInt(int *i, int size = 1);
    void marshalByte(char *b, int size = 1); 
    void marshalChar(char *c, int size = 1);
    void marshalFloat(float *f, int size = 1);
    void marshalDouble(double *d, int size = 1);
    void marshalLong(long *l, int size = 1);
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
  };
}

#endif
