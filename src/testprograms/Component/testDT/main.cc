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
    cerr << "  -server  - server process\n";
    cerr << "  -client  - client process\n";
    cerr << "  -stop  - stop the server\n";
    cerr << "  -test  - test the ploader\n";
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

int main(int argc, char* argv[])
{
    using std::string;
    try{
      bool client=false;
      bool server=false;
      bool stop=false;
      bool test=false;
      string url;
      int reps=1;
      
      for(int i=1;i<argc;i++){
	string arg(argv[i]);
	if(arg == "server") server=true;
        else if(arg == "client") client=true;
        else if(arg == "stop")   stop=true;
        else if(arg == "test")   test=true;
        else url=arg;
      }
      if(!client && !server && !stop && !test)
	usage(argv[0]);


      PIDL::initialize(argc, argv);
      if(server) {
	DTPoint *ep=new DTPoint;
	ofstream f("pp.url");
	std::string s;
	f<<PIDL::getDT()->getUrl()<<endl;
	f<<(long)ep<<endl;
	f.close();
	/*	
	for(int i=0; i<20;i++){
	  sleep(5);
	  Thread* t = new Thread(new MyThread, "MyThread", 0, Thread::Activated);
	  t->detach();
	} 
	*/
	
	while(true){
	  DTMessage *msg=ep->getMessage();
	  cerr<<"getMessage="<<msg->buf<<endl;
	  DTMessage *rm=new DTMessage;
	  rm->buf="I got it";
	  rm->length=strlen(msg->buf);
	  rm->autofree=false;
	  rm->to_addr=msg->fr_addr;
	  rm->recver=msg->sender;
	  ep->putMessage(rm);
	}
      }
      else if(client){
	ifstream f("pp.url");
	DTPoint *sp=new DTPoint;
	long ep;
	string ppurl;
	f>>ppurl;
	f>>ep;
	cerr<<"ppurl="<<ppurl<<ep<<endl;
	f.close();

	URL url(ppurl);

	int port=url.getPortNumber();
	long ip=url.getIP();

	for(int i=0; i<5; i++){
	  sleep(2);
	  DTMessage *msg=new DTMessage;
	  msg->buf=new char[40];
	  sprintf(msg->buf, "$This is message #%d$",i);
	  msg->length=strlen(msg->buf);
	  msg->autofree=false;
	  msg->to_addr.port=port;
	  msg->to_addr.ip  =ip;
	  msg->recver= (DTPoint *)ep;
	  sp->putMessage(msg);
	  msg=sp->getMessage();
	  cerr<<"getMessage="<<msg->buf<<endl;	  
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





