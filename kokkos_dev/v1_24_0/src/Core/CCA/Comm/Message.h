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
    virtual int getRecvBufferCopy(void* buf) = 0;
    virtual int getSendBufferCopy(void* buf) = 0;
    virtual void setRecvBuffer(void* buf, int len) = 0;
    virtual void setSendBuffer(void* buf, int len) = 0;
    virtual void createMessage() = 0;
    virtual void marshalInt(const int *i, int size = 1) = 0;
    virtual void marshalSpChannel(SpChannel* channel) = 0;
    virtual void marshalByte(const char *b, int size = 1) = 0; 
    virtual void marshalChar(const char *c, int size = 1) = 0;
    virtual void marshalFloat(const float *f, int size = 1) = 0;  
    virtual void marshalDouble(const double *d, int size = 1) = 0;
    virtual void marshalLong(const long *l, int size = 1) = 0;
    virtual void marshalOpaque(void **ptr, int size = 1) = 0;

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
    virtual void unmarshalOpaque(void **ptr, int size = 1) = 0;
    virtual void destroyMessage() = 0;

  };
}

#endif
