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

#include "SocketMessage.h"
using namespace SCIRun;

SocketMessage::SocketMessage() { 
}

SocketMessage::~SocketMessage() {
}

void SocketMessage::createMessage()  {  }
void SocketMessage::marshalInt(const int *i, int size)  {  }
void SocketMessage::marshalByte(const char *b, int size) {  }
void SocketMessage::marshalChar(const char *c, int size) {  }
void SocketMessage::marshalFloat(const float *f, int size) {  }
void SocketMessage::marshalDouble(const double *d, int size) {  }
void SocketMessage::marshalLong(const long *l, int size) {  }
void SocketMessage::marshalSpChannel(SpChannel* channel) {  }
void SocketMessage::sendMessage(int handler) {  }
void SocketMessage::waitReply()  {  }
void SocketMessage::unmarshalReply()  {  }
void SocketMessage::unmarshalInt(int *i, int size) {  }
void SocketMessage::unmarshalByte(char *b, int size) {  }
void SocketMessage::unmarshalChar(char *c, int size) {  }
void SocketMessage::unmarshalFloat(float *f, int size) {  }
void SocketMessage::unmarshalDouble(double *d, int size) {  }
void SocketMessage::unmarshalLong(long *l, int size) {  }
void SocketMessage::unmarshalSpChannel(SpChannel* channel) {  }

void* SocketMessage::getLocalObj() {  return 0; }

void SocketMessage::destroyMessage() {  }
