
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
::std::cerr << "relayException START\n";
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
::std::cerr << "0relayException THROW " << *x_id << "\n";
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
::std::cerr << "1relayException THROW " << *x_id << "\n";
        xdb.clear();
	return;
      }
      xiter++;
    }
    
    //Now read for the real exception
    while(realxID != readException(message,&xlineID)) sleep(1);
    (*x_id) = realxID;   
::std::cerr << "2relayException THROW " << *x_id << "\n";
    xdb.clear();
    return;
  }

  assert(false);
}

int XceptionRelay::checkException(Message** _xMsg) 
{ 
::std::cerr << "checkException START\n";
  int xlineID=1;
  int xID;

  //Increment lineID
  lineID++;

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
::std::cerr << "0checkException THROW " << xID << "\n";
        return (xID);
      }
      else {
	//Our exception was not elected!!!!
	//Check XDB for the real exception
	XDB::iterator xiter = xdb.begin();
	while(xiter != xdb.end()) {
	  if(((*xiter).second.xID) == realxID) {
	    (*_xMsg) = (*xiter).second.xMsg;
::std::cerr << "1checkException THROW " << realxID << "\n";
	    return realxID;
	  }
	  xiter++;
	}
	
	//Now read for the real exception
	while(realxID != readException(_xMsg,&xlineID)) sleep(1);
::std::cerr << "2checkException THROW " << realxID << "\n";
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
