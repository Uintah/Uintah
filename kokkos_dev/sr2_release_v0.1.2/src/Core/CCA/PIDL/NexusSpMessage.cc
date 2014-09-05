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



#include <Core/CCA/PIDL/NexusSpMessage.h>
#include <Core/CCA/PIDL/ReplyEP.h>
#include <Core/CCA/PIDL/NexusSpChannel.h>
#include <Core/CCA/Exceptions/CommError.h>
#include <iostream>

using namespace SCIRun;

#define BUF_SIZE 10000

void NexusSpMessage::printDebug(const std::string& d) {
  std::cout << d << endl;
}

NexusSpMessage::NexusSpMessage(globus_nexus_startpoint_t* sp) {
  //Get startpoint from Channel
  d_sp = sp;
  _buffer = 0;
  _recvbuff = 0;
}

NexusSpMessage::~NexusSpMessage() {  }

void NexusSpMessage::createMessage() {
  if (kDEBUG) printDebug("NexusSpMessage::createMessage()");

  //Setting up reply
  _reply= ReplyEP::acquire();
  _reply->get_startpoint_copy(&_reply_sp);
  //Record the size of the reply startpoint
  msgsize = globus_nexus_sizeof_startpoint(&_reply_sp,1);
  //Creating the message buffer
  _buffer = new globus_nexus_buffer_t();
  if(int _gerr=globus_nexus_buffer_init(_buffer, BUF_SIZE, 0))
    throw CommError("buffer_init", __FILE__, __LINE__, _gerr);
}

void NexusSpMessage::marshalInt(const int *i, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_int(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_int(_buffer, (int*)i, size);
}

void NexusSpMessage::marshalByte(const char *b, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  globus_byte_t* gb = (globus_byte_t*) b;
  int s = globus_nexus_sizeof_byte(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_byte(_buffer, gb, size);
}

void NexusSpMessage::marshalChar(const char *c, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_char(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_char(_buffer, (char*)c, size);
}

void NexusSpMessage::marshalFloat(const float *f, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_float(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_float(_buffer, (float*)f, size);
}

void NexusSpMessage::marshalDouble(const double *d, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_double(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_double(_buffer, (double*)d, size);
}

void NexusSpMessage::marshalLong(const long *l, int size) {
  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  int s = globus_nexus_sizeof_long(size);
  msgsize += s;
  globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
  globus_nexus_put_long(_buffer, (long*)l, size);
}

void NexusSpMessage::marshalSpChannel(SpChannel* channel) {
  if (kDEBUG) printDebug("NexusSpMessage::marshalSpChan()");

  if (!_buffer)
    throw CommError("uninitialized buffer on marshaling call", __FILE__, __LINE__, 1000);
  NexusSpChannel * nex_chan = dynamic_cast<NexusSpChannel*>(channel);
  if (nex_chan) {
    int s = globus_nexus_sizeof_startpoint(&(nex_chan->d_sp),1);
    msgsize += s;
    globus_nexus_check_buffer_size(_buffer,s,BUF_SIZE,0,0);
    globus_nexus_put_startpoint_transfer(_buffer, &(nex_chan->d_sp), 1);
  }
  else {
    throw CommError("Communication library discrepancy detected", __FILE__, __LINE__, 1001);
  }
}

void NexusSpMessage::sendMessage(int handler) {
  if (kDEBUG) printDebug("NexusSpMessage::sendMessage()");

  if (!_buffer)
    throw CommError("uninitialized buffer on send message call", __FILE__, __LINE__, 1000);
  // Marshal the reply startpoint
  globus_nexus_put_startpoint_transfer(_buffer, &_reply_sp, 1);
  // Send the message
  if(int _gerr=globus_nexus_send_rsr(_buffer, d_sp,
				     handler, GLOBUS_TRUE, GLOBUS_FALSE))
    throw CommError("send_rsr", __FILE__, __LINE__, _gerr);
  delete _buffer;
  _buffer = 0;
}

void NexusSpMessage::waitReply() {
  _recvbuff = new globus_nexus_buffer_t();
  (*_recvbuff) = _reply->wait();
}

void NexusSpMessage::unmarshalReply() { }

void NexusSpMessage::unmarshalInt(int* i, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_int(_recvbuff, i, size);
}

void NexusSpMessage::unmarshalByte(char *b, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_byte_t gb;
  globus_nexus_get_byte(_recvbuff, &gb, size);
  *b = (char) gb;
}

void NexusSpMessage::unmarshalChar(char *c, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_char(_recvbuff, c, size);
}

void NexusSpMessage::unmarshalFloat(float *f, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_float(_recvbuff, f, size);
}

void NexusSpMessage::unmarshalDouble(double *d, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_double(_recvbuff, d, size);
}

void NexusSpMessage::unmarshalLong(long* l, int size) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  globus_nexus_get_long(_recvbuff, l, size);
}

void NexusSpMessage::unmarshalSpChannel(SpChannel* channel) {
  if (!_recvbuff)
    throw CommError("empty reply buffer on unmarshaling call", __FILE__, __LINE__, 1000);
  NexusSpChannel* nex_chan = dynamic_cast<NexusSpChannel*>(channel);
  if (!nex_chan)
    throw CommError("error in comm. libraries in unmarshaling call", __FILE__, __LINE__, 1000);
  if(int _gerr=globus_nexus_get_startpoint(_recvbuff,&(nex_chan->d_sp), 1))
    throw CommError("get_startpoint",_gerr);
}

void* NexusSpMessage::getLocalObj() {
  if (!d_sp) {
    throw CommError("d_sp = NULL (getLocalObj)", __FILE__, __LINE__, 1000);
  }

  if(globus_nexus_startpoint_to_current_context(d_sp)){
    globus_nexus_endpoint_t *ep;
    if(int _gerr=globus_nexus_startpoint_get_endpoint(d_sp, &ep))
      throw CommError("get_endpoint", __FILE__, __LINE__, _gerr);
    return (globus_nexus_endpoint_get_user_pointer(ep));
  }
  else {
    return 0;
  }
}

void NexusSpMessage::destroyMessage() {
  if (_recvbuff) {
    if(int _gerr=globus_nexus_buffer_destroy(_recvbuff))
      throw CommError("buffer_destroy", __FILE__, __LINE__, _gerr);
    delete _recvbuff;
    _recvbuff = 0;
  }
  ReplyEP::release(_reply);
  delete this;
}

int NexusSpMessage::getRecvBufferCopy(void* buf)
{
  //  globus_nexus_get_user(_recvbuff, (globus_byte_t *)buf, 50);
  return 50;
}

int NexusSpMessage::getSendBufferCopy(void* buf)
{
  //  globus_nexus_get_user(_buffer, (globus_byte_t *)buf, msgsize);
  return msgsize;
}

void NexusSpMessage::setRecvBuffer(void* buf, int len)
{
  //  globus_nexus_put_user(_recvbuff, (globus_byte_t *)buf, len);
}

void NexusSpMessage::setSendBuffer(void* buf, int len)
{
  //  globus_nexus_put_user(_buffer, (globus_byte_t *)buf, len);
}

void NexusSpMessage::marshalOpaque(void **buf, int size)
{
}
void NexusSpMessage::unmarshalOpaque(void **buf, int size)
{
}
