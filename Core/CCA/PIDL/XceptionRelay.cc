
#include <Core/CCA/PIDL/XceptionRelay.h>
#include <iostream>
#include <string.h> 
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

void XceptionRelay::relayException(int x_id, Message* message) 
{ 
  IntraComm* icomm = mypb->rm.intracomm; 
  //If this is a serial object, intracomm. may be NULL
  if(icomm == NULL) return;

  ////////
  // Message format
  // [length][x_id][lineId][MESSAGE]

  //Marshal all the necessary data in the buffer
  int length = 0;
  char* p_sbuf = sbuf + sizeof(int);
  (*p_sbuf) = x_id;
  p_sbuf += sizeof(int);
  length += sizeof(int);
  (*p_sbuf) = lineID;
  p_sbuf += sizeof(int);
  length += sizeof(int);
  length += (message->getRecvBufferCopy(p_sbuf));
  (*sbuf) = length;
  ::std::cout << "Packed Message\n";

  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      icomm->async_send(k,sbuf,MAX_X_MSG_LENGTH);
      ::std::cout << "SENDING to " << k << "\n";
    } 
  }
}

void XceptionRelay::checkException() 
{ 
  IntraComm* icomm = mypb->rm.intracomm;
  Message* message = mypb->rm.getIndependentReference()->chan->getMessage();

  //If this is a serial object, intracomm. may be NULL
  if(icomm == NULL) return;

  //Increment lineID
  lineID++;

  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      if(icomm->async_receive(k,rbuf,MAX_X_MSG_LENGTH)) {
	//Unmarshal data and store it in the XDB
	Xception newX;
	char* p_rbuf = rbuf;
	int length = (int)(*p_rbuf);
	assert(length <= MAX_X_MSG_LENGTH);
	p_rbuf += sizeof(int);
	newX.xID = (int)(*p_rbuf);
	p_rbuf += sizeof(int);	
	int xlineID = (int)(*p_rbuf);
	p_rbuf += sizeof(int);
	length -= (3*sizeof(int));
	message->setRecvBuffer(p_rbuf,length);
	newX.xMsg = message;
	::std::cerr << "New exception; x_id=" << newX.xID << ", lineID=" << lineID << ", xlineID=" << xlineID << "\n";
	xdb.insert(XDB_valType(xlineID, newX));
      }
    }
  }



  //Check XDB for exceptions'need throwin'

}


