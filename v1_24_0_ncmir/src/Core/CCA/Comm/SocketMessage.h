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
#include <Core/CCA/Comm/DT/DTMessageTag.h>

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
    void marshalOpaque(void **ptr, int size = 1);

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
    void unmarshalOpaque(void** ptr, int size = 1);
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
    DTMessageTag tag;
  };
}

#endif
