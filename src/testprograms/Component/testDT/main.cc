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


/*
 *  pingpong.cc
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

using namespace SCIRun;
using namespace std;


const int SIZE=40; //80000; //maximum message size
const int nServer=5; //number of servers in each process
const int nMsg=10;  //number of messages sent from one point to another point.

//this method creates a message body (id+message), 
//or does error check
void create_check_message(int id, DTMessage *msg, bool errcheck=false){
  int len;

  if(errcheck){
    len=msg->length;
    for(int i=sizeof(int); i<len; i++){
      if(msg->buf[i]!=char((id+i)%128)){
	cerr<<"Error: wrong message body"<<endl;
      }
    }
  }
  else{
    len= ((unsigned)rand())%SIZE;
    if(len<(int)sizeof(int)) len=sizeof(int);
    msg->buf=new char[len];
    *((int*)msg->buf)=id;
    for(int i=sizeof(int); i<len; i++){
      msg->buf[i]=char((id+i)%128);
    }
    msg->length=len;
  }
}

class EPThread : public Runnable{
public:
  EPThread(DTAddress *addrlist, DTPoint **eplist, DTPoint *me, int mpi_size, bool isSender, Semaphore *sema){
    this->addrlist=addrlist;
    this->eplist=eplist;
    this->me=me;
    this->mpi_size=mpi_size;
    this->isSender=isSender;
    this->sema=sema;
      
  }
  
  void run(){
    srand(clock());
    
    //sending log:
    //from me to mpi_size*nServer
    
    //recving log
    //from mpi_size*nServer to me
    
    int *log=new int[mpi_size*nServer];
    for(int i=0; i<mpi_size*nServer; i++) log[i]=0;
    if(isSender){
      //cerr<<"Sending thread is working"<<endl;
      while(true){
	bool done=true;
	for(int i=0; i<mpi_size*nServer; i++){
	  if(log[i]<nMsg){
	    done=false;
	    break;
	  }
	}
	if(done) break;

	int iaddr = ((unsigned)rand())%mpi_size; 
	int ipt = nServer*iaddr+((unsigned)rand())%nServer;

	//update the sender's log
	if(log[ipt]>=nMsg) continue;
	//cerr<<"send message to : iaddr/ipt/log[ipt]="<<iaddr<<"/"<<ipt<<"/"<<log[ipt]<<endl;

	
	DTMessage *msg=new DTMessage;
	create_check_message(log[ipt], msg);
	msg->recver=eplist[ipt];
	msg->to_addr=addrlist[iaddr];
	msg->autofree=true;
	//cerr<<"Send a message:"<<endl;
	//msg->display();
	me->putMessage(msg);

	log[ipt]++;
	//if(iter++%10==0)sleep(1); 
      }
      cerr<<"Sending job is done"<<endl;
      sema->up();
    }
    else{
      //cerr<<"Receiving thread is working"<<endl;
      while(true){
	bool done=true;
	for(int i=0; i<mpi_size*nServer; i++){
	  if(log[i]<nMsg){
	    done=false;
	    break;
	  }
	}
	if(done) break;

	DTMessage *msg=me->getMessage();
	bool bad=true;
	int iaddr=-1;
	for(iaddr=0; iaddr<mpi_size; iaddr++){
	  if(addrlist[iaddr]==msg->fr_addr){
	    bad=false;
	    break;
	  }
	}

	if(bad) cerr<<"Error: unexpected from address"<<endl;
	bad=true;
	int ipt;
	for(ipt=iaddr*nServer; ipt<(iaddr+1)*nServer;ipt++){
	  if(eplist[ipt]==msg->sender){
	    bad=false;
	    break;
	  }
	}
	if(bad) cerr<<"Error: unexpected sender"<<endl;

	int id=*((int*)(msg->buf));
	if(id!=log[ipt]){
	  cerr<<"Error: received message in bad order."<<endl;
	  cerr<<"id/log[ipt]="<<id<<"/"<<log[ipt]<<endl;
	}

	create_check_message(id, msg, true); //message body error check

	log[ipt]++;

	//cerr<<"get a message"<<endl;
	//msg->display();
	delete msg;
      }
      cerr<<"Recving job is done"<<endl;
      sema->up();
    }
    delete []log;
  }
private:
  DTPoint *me;
  DTAddress *addrlist;
  DTPoint **eplist;
  int mpi_size;
  bool isSender;
  Semaphore *sema;
};


int main(int argc, char* argv[])
{
  Semaphore *sema;
  
  using std::string;
  try{
    MPI_Init(&argc,&argv);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    PIDL::initialize(mpi_rank, mpi_size);

    sema=new Semaphore("dt points semapore", 0);

    DTPoint *ep[nServer];
    for(int i=0; i<nServer; i++){
      ep[i]=new DTPoint(PIDL::getDT());
    }
      
    DTAddress *addrlist=new DTAddress[mpi_size];
    
    DTAddress myaddr;
    URL url(PIDL::getDT()->getUrl());
    myaddr.ip= url.getIP();
    myaddr.port= url.getPortNumber();
    
    MPI_Allgather(&myaddr, sizeof(DTAddress), MPI_CHAR,  
		  addrlist, sizeof(DTAddress), MPI_CHAR,   MPI_COMM_WORLD);
    
    if(mpi_rank==0){
	cerr<<"gathered addresses are:"<<endl;
	for(int i=0; i<mpi_size; i++){
	  cerr<<"addr "<<i<<"="<<addrlist[i].ip<<":"<<addrlist[i].port<<endl;
	}
    }
    
    DTPoint **eplist=new (DTPoint*)[nServer*mpi_size];
    
    MPI_Allgather(ep, sizeof(DTPoint *)*nServer, MPI_CHAR,   
		  eplist, sizeof(DTPoint *)*nServer, MPI_CHAR,  MPI_COMM_WORLD);
    
    if(mpi_rank==0){
      cerr<<"gathered eps are:"<<endl;
      for(int i=0; i<mpi_size; i++){
	cerr<<"rank="<<i<<endl;
	for(int j=0; j<nServer; j++){
	  cerr<<"ep"<<j<<"="<<eplist[i*nServer+j]<<endl;
	}
      }
    }
    
    
    // start one thread for each ep, so that it can
    // randomly send messages.
    for(int i=0; i<nServer; i++){
      Thread *ep_thread= new Thread(new EPThread(addrlist, eplist, ep[i], mpi_size, true, sema), "ep sender thread", 0, Thread::Activated);
	ep_thread->detach();
    }
    
    // start one thread for each ep, so that it can
    // check and receive the incoming messages.
    for(int i=0; i<nServer; i++){
      Thread *ep_thread= new Thread(new EPThread(addrlist, eplist, ep[i], mpi_size, false, sema), "ep receiver thread", 0, Thread::Activated);
      ep_thread->detach();
    }

    for(int i=0; i<nServer*2;i++){
      sema->down();
    }
    delete sema;
    cerr<<"testDT done on rank="<<mpi_rank<<endl;
  }catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }
  
  PIDL::finalize();
  MPI_Finalize();
  return 0;
}





