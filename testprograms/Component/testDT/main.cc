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
#include <Core/CCA/Comm/DT/DTThread.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/Thread/Thread.h>

using namespace SCIRun;
using namespace std;


using namespace std;
using namespace SCIRun;



void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  server  - server process\n";
    cerr << "  client  - client process\n";
    cerr << "\n";
    exit(1);
}


  
class MyThread :public Runnable{
public:
  void run()
  {
    cerr<<"One thread starts...\n";
    while(1){
      //sleep(1);
    }
  }
  
};

const int nServer=10;
const int nClient=10;
const int nRep=100;
const int SIZE=80000;

int main(int argc, char* argv[])
{
    using std::string;
    try{
      bool client=false;
      bool server=false;
      bool stop=false;
      bool test=false;
      string url;
      
      for(int i=1;i<argc;i++){
	string arg(argv[i]);
	if(arg == "server") server=true;
        else if(arg == "client") client=true;
        else url=arg;
      }
      if(!client && !server && !stop && !test)
	usage(argv[0]);


      PIDL::initialize();
      if(server) {
	DTPoint *ep[nServer];
	ofstream f("pp.url");
	f<<PIDL::getDT()->getUrl()<<endl;
	for(int i=0; i<nServer; i++){
	  ep[i]=new DTPoint(PIDL::getDT());
	  f<<(long)ep[i]<<endl;
	}
	f.close();

	while(true){
	  for(int i=0; i<nServer; i++){
	    DTMessage *msg=ep[i]->getMessage();
	    cerr<<"getMessage="<<msg->buf<<endl;
	    /*
	    DTMessage *rm=new DTMessage;
	    rm->buf="I got it";
	    rm->length=strlen(msg->buf);
	    rm->autofree=false;
	    rm->to_addr=msg->fr_addr;
	    rm->recver=msg->sender;
	    ep->putMessage(rm);
	    */
	  }
	}
      }
      else if(client){
	ifstream f("pp.url");
	DTPoint *sp[nClient];

	for(int i=0; i<nClient; i++){
	  sp[i]=new DTPoint(PIDL::getDT());
	}

	cerr<<"sender="<< (long)sp<<endl;
	long ep[nServer];
	string ppurl;
	f>>ppurl;
	for(int i=0; i<nServer; i++){
	  f>>ep[i];
	  cerr<<"ppurl="<<ppurl<<ep[i]<<endl;
	}
	f.close();

	URL url(ppurl);

	int port=url.getPortNumber();
	long ip=url.getIP();

	for(int k=0; k<nRep; k++){
	  for(int j=0; j<nClient; j++){
	    for(int i=0; i<nServer; i++){
	      DTMessage *msg=new DTMessage;
	      msg->buf=new char[SIZE];
	      msg->length=SIZE;
	      sprintf(msg->buf, "$This is LONG message #%d from client %d to server %d $...",k, j, i);
	      msg->autofree=true;
	      msg->to_addr.port=port;
	      msg->to_addr.ip  =ip;
	      msg->recver= (DTPoint *)ep[i];
	      sp[j]->putMessage(msg);
	      //msg=sp[j]->getMessage();
	      //cerr<<"getMessage="<<msg->buf<<endl;	  
	    }
	  }
	}
      }
    } catch(const MalformedURL& e) {
      cerr << "pp.cc: Caught MalformedURL exception:\n";
      cerr << e.message() << '\n';
    } catch(...) {
      cerr << "Caught unexpected exception!\n";
      abort();
    }
    PIDL::finalize();
    return 0;
}





