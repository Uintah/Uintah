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
void SocketMessage::marshalInt(int *i, int size = 1)  {  }
void SocketMessage::marshalByte(char *b, int size = 1) {  }
void SocketMessage::marshalChar(char *c, int size = 1) {  }
void SocketMessage::marshalFloat(float *f, int size = 1) {  }
void SocketMessage::marshalDouble(double *d, int size = 1) {  }
void SocketMessage::marshalLong(long *l, int size = 1) {  }
void SocketMessage::marshalSpChannel(SpChannel* channel) {  }
void SocketMessage::sendMessage(int handler) {  }
void SocketMessage::waitReply()  {  }
void SocketMessage::unmarshalReply()  {  }
void SocketMessage::unmarshalInt(int *i, int size = 1) {  }
void SocketMessage::unmarshalByte(char *b, int size = 1) {  }
void SocketMessage::unmarshalChar(char *c, int size = 1) {  }
void SocketMessage::unmarshalFloat(float *f, int size = 1) {  }
void SocketMessage::unmarshalDouble(double *d, int size = 1) {  }
void SocketMessage::unmarshalLong(long *l, int size = 1) {  }
void SocketMessage::unmarshalSpChannel(SpChannel* channel) {  }

void* SocketMessage::getLocalObj() {  return 0; }

void SocketMessage::destroyMessage() {  }
