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


#ifndef MESSAGE_INTERFACE_H
#define MESSAGE_INTERFACE_H 

namespace SCIRun {
  class SpChannel;

  /**************************************
 
  CLASS
     Message
   
  DESCRIPTION
    The base class for all communication-specific message 
    abstractions. This class describes the methods one
    communication instance needs to implement in order
     to work properly under the sidl compiler. The names
     of the methods themselver are self-explanatory and
     need no further clarification.

  ****************************************/

  class Message {
  public:
    virtual ~Message();
    virtual void* getLocalObj() = 0;
    virtual void createMessage() = 0;
    virtual void marshalInt(const int *i, int size = 1) = 0;
    virtual void marshalSpChannel(SpChannel* channel) = 0;
    virtual void marshalByte(const char *b, int size = 1) = 0; 
    virtual void marshalChar(const char *c, int size = 1) = 0;
    virtual void marshalFloat(const float *f, int size = 1) = 0;  
    virtual void marshalDouble(const double *d, int size = 1) = 0;
    virtual void marshalLong(const long *l, int size = 1) = 0;
    virtual void sendMessage(int handler) = 0;
    virtual void waitReply() = 0;
    virtual void unmarshalReply() = 0;
    virtual void unmarshalInt(int* i, int size = 1) = 0;
    virtual void unmarshalByte(char* b, int size = 1) = 0;
    virtual void unmarshalChar(char *c, int size = 1) = 0;
    virtual void unmarshalFloat(float *f, int size = 1) = 0;
    virtual void unmarshalDouble(double *d, int size = 1) = 0;
    virtual void unmarshalLong(long *l, int size = 1) = 0;
    virtual void unmarshalSpChannel(SpChannel* channel) = 0;
    virtual void destroyMessage() = 0;
  };
}

#endif
