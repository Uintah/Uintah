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



#include "NexusEpMessage.h"
#include <Core/CCA/Exceptions/CommError.h>
#include <Core/CCA/PIDL/ReplyEP.h>
#include <Core/CCA/PIDL/NexusSpChannel.h>
#include <iostream>

using namespace SCIRun;

#define BUF_SIZE 10000

void NexusEpMessage::printDebug(const std::string& d) {
  std::cout << d << endl;
}

NexusEpMessage::NexusEpMessage(globus_nexus_endpoint_t ep, globus_nexus_buffer_t msgbuff) {
  d_ep = ep;
  _buffer = msgbuff;
  _sendbuff = NULL;
  _reply = NULL;
  globus_nexus_startpoint_set_null(&msg_sp);
}

NexusEpMessage::~NexusEpMessage() { }

void NexusEpMessage::unmarshalReply() {
  if (kDEBUG) printDebug("NexusEpMessage::unmarshalReply()");
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  if(int _gerr=globus_nexus_get_startpoint(&_buffer,&msg_sp, 1))
    throw CommError("get_startpoint", __FILE__, __LINE__, _gerr);
}

void NexusEpMessage::unmarshalChar(char* c,int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_char(&_buffer, c, size);
}

void NexusEpMessage::unmarshalInt(int* i, int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_int(&_buffer, i, size);
}

void NexusEpMessage::unmarshalByte(char* b, int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_byte_t gb;
  globus_nexus_get_byte(&_buffer, &gb, size);
  *b = (char) gb;
}

void NexusEpMessage::unmarshalFloat(float* f, int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_float(&_buffer, f, size);
}

void NexusEpMessage::unmarshalDouble(double *d, int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_double(&_buffer, d, size);
}

void NexusEpMessage::unmarshalLong(long* l, int size) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_long(&_buffer, l, size);
}

void NexusEpMessage::unmarshalSpChannel(SpChannel* channel) {
  if (!_buffer)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  NexusSpChannel* nex_chan = dynamic_cast<NexusSpChannel*>(channel);
  if (!nex_chan)
    throw CommError("error in comm. libraries in unmarshaling call", __FILE__, __LINE__, 1000);
  if(int _gerr=globus_nexus_get_startpoint(&_buffer,&(nex_chan->d_sp), 1))
    throw CommError("get_startpoint", __FILE__, __LINE__, _gerr);
}

void* NexusEpMessage::getLocalObj() {
  return (globus_nexus_endpoint_get_user_pointer(&d_ep));
}

void NexusEpMessage::createMessage() {
  if (kDEBUG) printDebug("NexusEpMessage::createMessage()");

  //Creating the message buffer
  _sendbuff = new globus_nexus_buffer_t();
  if(int _gerr=globus_nexus_buffer_init(_sendbuff, BUF_SIZE, 0))
    throw CommError("buffer_init", __FILE__, __LINE__, _gerr);
  //Setting up reply
  _reply= ReplyEP::acquire();
  _reply->get_startpoint_copy(&_reply_sp);
  //Record the size of the reply startpoint
  msgsize = globus_nexus_sizeof_startpoint(&_reply_sp,1);
}

void NexusEpMessage::marshalInt(const int *i, int size) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_int(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_int(_sendbuff,(int*) i, size);
}


void NexusEpMessage::marshalByte(const char *b, int size) {
  if (!_sendbuff)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  globus_byte_t* gb = (globus_byte_t*) b;
  int s = globus_nexus_sizeof_byte(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_byte(_sendbuff, gb, size);
}

void NexusEpMessage::marshalSpChannel(SpChannel* channel) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  NexusSpChannel * nex_chan = dynamic_cast<NexusSpChannel*>(channel);
  if (nex_chan) {
    int s = globus_nexus_sizeof_startpoint(&(nex_chan->d_sp),1);
    msgsize += s;
    globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
    globus_nexus_put_startpoint_transfer(_sendbuff, &(nex_chan->d_sp), 1);
  } else {
    throw CommError("Communication library discrepancy detected", __FILE__, __LINE__, 1001);
  }
}

void NexusEpMessage::marshalChar(const char *c, int size) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_char(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_char(_sendbuff, (char*)c, size);
}

void NexusEpMessage::marshalFloat(const float *f, int size) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_float(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_float(_sendbuff, (float*)f, size);
}

void NexusEpMessage::marshalDouble(const double *d, int size) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_double(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_double(_sendbuff, (double*)d, size);
}

void NexusEpMessage::marshalLong(const long *l, int size) {
  if (!_sendbuff)
    throw CommError("trying to marshal to an unintialized buffer", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_long(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_sendbuff,s,BUF_SIZE,0,0);
  globus_nexus_put_long(_sendbuff, (long*)l, size);
}

void NexusEpMessage::sendMessage(int handler) {
  if (kDEBUG) printDebug("NexusEpMessage::sendMessage()");

  if (!_sendbuff)
    throw CommError("trying to send a message containing an unintialized buffer", __FILE__, __LINE__, 1000);
  if (globus_nexus_startpoint_is_null(&msg_sp))
    throw CommError("trying to send a message with and uninitialized sp", __FILE__, __LINE__, 1000);

  // Marshal the reply startpoint
  globus_nexus_put_startpoint_transfer(_sendbuff, &_reply_sp, 1);
  // Send the message
  if(int _gerr=globus_nexus_send_rsr(_sendbuff, &msg_sp,
				     handler, GLOBUS_TRUE, GLOBUS_FALSE))
    throw CommError("send_rsr", __FILE__, __LINE__, _gerr);
  delete _sendbuff;
  _sendbuff = NULL;
}

void NexusEpMessage::waitReply() { }

void NexusEpMessage::destroyMessage() {
  if (kDEBUG) printDebug("NexusEpMessage::destroyMessage()");
  if(int _gerr=globus_nexus_buffer_destroy(&_buffer))
    throw CommError("buffer_destroy", __FILE__, __LINE__, _gerr);
  if (_reply)
    ReplyEP::release(_reply);
  if (!globus_nexus_startpoint_is_null(&msg_sp)) {
    if(int _gerr=globus_nexus_startpoint_eventually_destroy(&msg_sp, GLOBUS_FALSE, 30))
      throw CommError("startpoint_eventually_destroy", __FILE__, __LINE__, _gerr);
  }
}

int NexusEpMessage::getRecvBufferCopy(void* buf)
{
  //  globus_nexus_get_user(_buffer, (globus_byte_t *)buf, 50);
  return 50;
}

int NexusEpMessage::getSendBufferCopy(void* buf)
{
  //  globus_nexus_get_user(_sendbuff, (globus_byte_t *)buf, msgsize);
  return msgsize;
}

void NexusEpMessage::setRecvBuffer(void* buf, int len)
{
  //  globus_nexus_put_user(_buffer, (globus_byte_t *)buf, len);
}

void NexusEpMessage::setSendBuffer(void* buf, int len)
{
  //  globus_nexus_put_user(_sendbuff, (globus_byte_t *)buf, len);
}

void NexusSpMessage::marshalOpaque(void **buf, int size)
{
}
void NexusSpMessage::unmarshalOpaque(void **buf, int size)
{
}
