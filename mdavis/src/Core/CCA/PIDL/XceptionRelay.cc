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


#include <Core/CCA/PIDL/XceptionRelay.h>
#include <iostream>
#include <string.h> 
#include <unistd.h>
#include <assert.h>

using namespace SCIRun;

XceptionRelay::XceptionRelay(ProxyBase* pb) 
  : lineID(1), mypb(pb)
{ 
}

XceptionRelay::~XceptionRelay() 
{ 
  //Clean up the map?

  //Sending buffer has to stay until all sends are finished
}

void XceptionRelay::relayException(int* x_id, Message** message) 
{ 
  IntraComm* icomm = mypb->rm.intracomm; 
  int xlineID;

  //If this is a serial object, intracomm. may be NULL
  if(icomm == NULL) return;

  ////////
  // Message format
  // [length][x_id][lineId][MESSAGE]

  //Marshal all the necessary data in the buffer
  int length = 0;
  char* p_sbuf = sbuf + sizeof(int);
  memcpy(p_sbuf,x_id,sizeof(int));
  p_sbuf += sizeof(int);
  length += sizeof(int);
  memcpy(p_sbuf,&lineID,sizeof(int));
  p_sbuf += sizeof(int);
  length += sizeof(int);
  length += ((*message)->getRecvBufferCopy(p_sbuf));
  memcpy(sbuf,&length,sizeof(int));

  //Send to all cohorts
  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      icomm->async_send(k,sbuf,MAX_X_MSG_LENGTH);
    } 
  }
  
  //Synchronize before throwing
  int realxID = mypb->_proxygetException(*x_id,lineID);  

  lineID=1;

  //Throw
  if(*x_id == realxID) {
    xdb.clear();
    return;
  }
  else {
    //Our exception was not elected!!!!
    //Check XDB for the real exception
    XDB::iterator xiter = xdb.begin();
    while(xiter != xdb.end()) {
      if(((*xiter).second.xID) == realxID) {
	(*message) = (*xiter).second.xMsg;
	(*x_id) = realxID;
        xdb.clear();
	return;
      }
      xiter++;
    }
    
    //Now read for the real exception
    while(realxID != readException(message,&xlineID)) sleep(1);
    (*x_id) = realxID;   
    xdb.clear();
    return;
  }

  assert(false);
}

int XceptionRelay::checkException(Message** _xMsg) 
{ 
  int xlineID=1;
  int xID;

  //Increment lineID
  lineID++;

  //Exit if proxy not parallel
  if(mypb->rm.getSize() == 1) return 0;

  //Check XDB for exceptions'need throwin'
  XDB::iterator xiter;
  xiter = xdb.find(lineID);
  if(xiter != xdb.end()) {
    (*_xMsg) = (*xiter).second.xMsg;
    xID = ((*xiter).second.xID);
  } else { 
    //Read for an exception
    xID = readException(_xMsg,&xlineID);
  }

  if(xID > 0) {
    //Throw right away if it is past due
    if(xlineID <= lineID) {
      //Synchronize before throwing
      int realxID = mypb->_proxygetException(xID,xlineID);

      //Reset lineID, we know we will throw something
      lineID=1;
      xdb.clear();

      //Throw
      if(xID == realxID) {
        return (xID);
      }
      else {
	//Our exception was not elected!!!!
	//Check XDB for the real exception
	XDB::iterator xiter = xdb.begin();
	while(xiter != xdb.end()) {
	  if(((*xiter).second.xID) == realxID) {
	    (*_xMsg) = (*xiter).second.xMsg;
	    return realxID;
	  }
	  xiter++;
	}
	
	//Now read for the real exception
	while(realxID != readException(_xMsg,&xlineID)) sleep(1);
	return (realxID);
      }
    } else {
      //Store data in exception database 
      Xception newX;
      newX.xID = xID;
      newX.xMsg = (*_xMsg);
      xdb.insert(XDB_valType(lineID, newX));
    }
  }

  return 0;
}


int XceptionRelay::readException(Message** _xMsg, int* _xlineID)
{
  IntraComm* icomm = mypb->rm.intracomm;
  Message* message = mypb->rm.getIndependentReference()->chan->getMessage();
  
  //If this is a serial object, intracomm. may be NULL
  assert(icomm != NULL);

  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      if(icomm->async_receive(k,rbuf,MAX_X_MSG_LENGTH)) {
	//Unmarshal data 
	char* p_rbuf = rbuf;
	int length = *(int *)(p_rbuf);
	assert(length <= MAX_X_MSG_LENGTH);
	p_rbuf += sizeof(int);
	int xID = *(int *)(p_rbuf);
	p_rbuf += sizeof(int);	
	(*_xlineID) = *(int *)(p_rbuf);
	p_rbuf += sizeof(int);
	length -= (3*sizeof(int));

	char* msgbuf = (char*)malloc(length); 
        memcpy(msgbuf,p_rbuf,length);
	message->setRecvBuffer(msgbuf,length);
        message->unmarshalInt(&xID);

	::std::cerr << "New exception; x_id=" << xID << ", lineID="
        	    << lineID << ", xlineID=" << *_xlineID << "\n";

	(*_xMsg) = message;
	return (xID);	
      }
    }
  }

  (*_xlineID)=0;
  return 0;
}

int XceptionRelay::getlineID() 
{
  return lineID;
}

void XceptionRelay::resetlineID()
{
  lineID=0;	
}
