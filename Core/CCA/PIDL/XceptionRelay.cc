
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

  //Send to all cohorts
  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      icomm->async_send(k,sbuf,MAX_X_MSG_LENGTH);
      ::std::cout << "SENDING to " << k << ", xid=" << x_id << "\n";
    } 
  }
}

int XceptionRelay::checkException(Message** _xMsg) 
{ 
  IntraComm* icomm = mypb->rm.intracomm;
  Message* message = mypb->rm.getIndependentReference()->chan->getMessage();

  //If this is a serial object, intracomm. may be NULL
  assert(icomm != NULL);

  //Increment lineID
  lineID++;

  for(int k=0; k < mypb->rm.getSize(); k++) {
    if(k !=  mypb->rm.getRank()) {
      if(icomm->async_receive(k,rbuf,MAX_X_MSG_LENGTH)) {
	//Unmarshal data 
	char* p_rbuf = rbuf;
	int length = (int)(*p_rbuf);
	assert(length <= MAX_X_MSG_LENGTH);
	p_rbuf += sizeof(int);
	int xID = (int)(*p_rbuf);
	p_rbuf += sizeof(int);	
	int xlineID = (int)(*p_rbuf);
	p_rbuf += sizeof(int);
	length -= (3*sizeof(int));

	char* msgbuf = (char*)malloc(length); 
        memcpy(msgbuf,p_rbuf,length);
	message->setRecvBuffer(msgbuf,length);
        message->unmarshalInt(&xID);

	//Throw right away if it is past due
	if(xlineID <= lineID) {
	  ::std::cerr << "New exception; x_id=" << xID << ", lineID="
	              << lineID << ", xlineID=" << xlineID << "\n";

	  (*_xMsg) = message;
	  return (xID);
	}
 
	//Store data in exception database 
	Xception newX;
        newX.xID = xID;
	newX.xMsg = message;
        xdb.insert(XDB_valType(xlineID, newX));


      } //EndIf msgReceived
    }
  }

  //Check XDB for exceptions'need throwin'
  

  return 0;
}


