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
 *  TxtBuilder.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#include <CCA/Components/TxtBuilder/TxtBuilder.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace SCIRun;
using namespace sci::cca;

static Semaphore* startup;

extern "C" sci::cca::Component::pointer make_SCIRun_TxtBuilder()
{
  return sci::cca::Component::pointer(new TxtBuilder());
}

TxtBuilder::TxtBuilder()
{
}

TxtBuilder::~TxtBuilder()
{
}

void TxtBuilder::setServices(const sci::cca::Services::pointer& services)
{
  //builderPort.setServices(services);
  //sci::cca::TypeMap::pointer props = services->createTypeMap();
  //myBuilderPort::pointer bp(&builderPort);
  //  services->addProvidesPort(bp,"builderPort","sci.cca.ports.BuilderPort", props);
  //services->registerUsesPort("builder", "sci.cca.ports.BuilderPort", props);

  svc=services;
  bs=pidl_cast<sci::cca::ports::BuilderService::pointer>
    (svc->getPort("cca.BuilderService"));
  if(bs.isNull()){
    cerr << "Fatal Error: cannot get BuilderService port\n";
    return;
  }

  cr=pidl_cast<sci::cca::ports::ComponentRepository::pointer>
    (svc->getPort("cca.ComponentRepository"));
  if(bs.isNull()){
    cerr << "Fatal Error: cannot get ComponentRepository port\n";
    return;
  }

  startup=new Semaphore("TxtBuilder Thread startup wait", 0);
  Thread* t = new Thread(this, "SCIRun Builder");
  t->detach();
  startup->down();
 
  //do not delelet the following line	
  //builder->registerServices(services);
}

enum TxtCmd{ 
  USE_CLUSTER,
  LIST_ALL, 
  LIST_COMPONENTS, 
  LIST_PORTS, 
  LIST_COMPATIBLE,
  LIST_CONNECTIONS,
  CREATE,
  CONNECT,
  GO,
  DISCONNECT,
  DESTROY,
  EXECUTE,
  LOAD,
  SAVE_AS,
  SAVE,
  INSERT,
  CLEAR,
  QUIT,
  HELP,
  BAD_COMMAND
};

char *cmddesc[]={
  "use cluster <user> <domain>: create slave framework on user@domain",
  "list all:                    list all available components",
  "list components:             list all active components",
  "list ports <comp>:           list ports of component <comp>",
  "list compatible <comp>:      list compatible ports for component <comp>",
  "list connections <comp>:     list connections of component <comp>",
  "create <comp>:               create component <comp>",		
  "connect <c1> <p1> <c2> <p2>: connect port <p1> (of component <c1>) \n\t\t\tand port <p2> (of component <c2>)",
  "go <comp>:                   start component <comp>",		
  "disconnect <comp><port>:     disconnect connection from <port> of component <comp>",
  "destroy <comp>:              destroy component <comp>",		
  "execute <file>:		execute batch <file>",
  "load <network>:              load <network>",
  "save as <network>:           save network as <network>",
  "save:                        save network",
  "insert <network>:            insert <network>",  
  "clear:                       clear network",
  "quit:                        quit this builder",
  "help:                        provide help information",
};


void TxtBuilder::run()
{
  startup->up();
  system("sleep 2");
  cout<<"\n\n\nTxtBuilder starts...\n";
  cout<<"Type help for commands usage\n";

  while(true){
    const int n=256;
    char cmdline[n];
    cout<<"\nSCIRun>";
    cin.getline(cmdline,n);
    if(exec_command(cmdline))break;
  }
}

bool TxtBuilder::exec_command(char cmdline[])
{
   string args[4];
   int cmd=parse(cmdline,args);
   if (cmd==USE_CLUSTER) use_cluster(args);
   else if (cmd==LIST_ALL) list_all(args);
   else if(cmd==LIST_COMPONENTS) list_components(args);
   else if(cmd==LIST_PORTS) list_ports(args);
   else if(cmd==LIST_COMPATIBLE) list_compatible(args);
   else if(cmd==LIST_CONNECTIONS) list_connections(args);
   else if(cmd==CREATE) create(args);
   else if(cmd==CONNECT) connect(args);
   else if(cmd==GO) go(args);
   else if(cmd==DISCONNECT) disconnect(args);
   else if(cmd==DESTROY) destroy(args);
   else if(cmd==EXECUTE) execute(args);
   else if(cmd==LOAD) load(args);
   else if(cmd==SAVE_AS) save_as(args);
   else if(cmd==SAVE) save(args);
   else if(cmd==INSERT) insert(args);
   else if(cmd==CLEAR) clear(args);
   else if(cmd==QUIT) return true; //quit(args);
   else if(cmd==HELP) help(args);
   else bad_command(args);
   return false;
}

int TxtBuilder::parse(string cmdline, string args[])
{
  int num_cmds=sizeof(cmddesc)/sizeof(char *);
  for(int i=0; i<num_cmds; i++){
    //ref is the command part
    string cs(cmddesc[i]);
    string ref=cs.substr(0, cs.find(":",0));
    int nargs=0;
    while(true){
      unsigned int pos=ref.rfind("<");
      if(pos==string::npos) break;
      nargs++;
      ref=ref.substr(0,pos);
    }
    ref=ref.substr(0,ref.size());
    if(cmdline.find(ref)==0){
      string cl=cmdline.substr(ref.size(), cmdline.size()-ref.size());
      for(int j=0; j<nargs; j++){
	while(cl.size()>0 && cl[0]==' ')
	  cl=cl.substr(1,cl.size()-1);
	if(cl.size()==0) return BAD_COMMAND;
	if(j==nargs-1) args[j]=cl;
	else{
	  unsigned int pos=cl.find(" ",0);
	  if(pos==string::npos) return BAD_COMMAND;
	  args[j]=cl.substr(0,pos);
	  cl=cl.substr(pos,cl.size()-pos);
	}
      }
      return i;
    }
  }
  return BAD_COMMAND;
}


void TxtBuilder::use_cluster(string args[])
{
  string user=args[0];
  string domain=args[1];
  string cmd="ssh ";
  cmd+="kzhang@qwerty.sci.utah.edu ~/SCIRun/cca-debug/ploader from.buzz.sci.utah.edu:00:11";
  //cmd+=user+"@"+domain+" ~/SCIRun/cca-debug/ploader from.buzz.sci.utah.edu:00:11";
  cout<<"Connecting to "<<user<<"@"<<domain<<"...\n";
  cout<<cmd<<endl;
  system(cmd.c_str());
}

void TxtBuilder::list_all(string args[])
{
  SSIDL::array1<ComponentClassDescription::pointer> cds=cr->getAvailableComponentClasses();
  cout<<"Available Component Classes:\n";
  for(unsigned i=0; i<cds.size();i++){
    cout<<"#"<<i<<":\t"<<cds[i]->getComponentClassName()<<endl;
  }
}

void TxtBuilder::list_components(string args[])
{
  SSIDL::array1<ComponentID::pointer> comps=bs->getComponentIDs();
  cout<<"Active Components:\n";
  for(unsigned i=0; i<comps.size();i++){
    cout<<"#"<<i<<":\t"<<comps[i]->getInstanceName()<<endl;
  }
}

void TxtBuilder::list_ports(string args[])
{
  ComponentID::pointer cid=bs->getComponentID(args[0]);
  SSIDL::array1<string> ppNames=bs->getProvidedPortNames(cid);
  SSIDL::array1<string> upNames=bs->getUsedPortNames(cid);
  cout<<"Provided Port Names:\n";
  for(unsigned i=0; i<ppNames.size();i++){
    cout<<"#"<<i<<":\t"<<ppNames[i]<<endl;
  }
  cout<<"Used Port Names:\n";
  for(unsigned i=0; i<upNames.size();i++){
    cout<<"#"<<i<<":\t"<<upNames[i]<<endl;
  }
}

void TxtBuilder::list_compatible(string args[])
{
  ComponentID::pointer cid=bs->getComponentID(args[0]);
  SSIDL::array1<string> ppNames=bs->getProvidedPortNames(cid);
  SSIDL::array1<string> upNames=bs->getUsedPortNames(cid);

  SSIDL::array1<ComponentID::pointer> cids=bs->getComponentIDs();
  cout<<"Provided Port Names:\n";
  for(unsigned i=0; i<ppNames.size();i++){
    cout<<"#"<<i<<":\t"<<ppNames[i]<<endl;
    for(unsigned j=0; j<cids.size(); j++){
      if(cids[j]==cid)continue;
      SSIDL::array1<string> cps=bs->getCompatiblePortList(cid,ppNames[i],cids[j]);
      for(unsigned k=0; k<cps.size(); k++){
	cout<<"\t|___ "<<cps[k]<<"\t("<<cids[j]->getInstanceName()<<")"<<endl;
      }
    }
  }
  cout<<"Used Port Names:\n";
  for(unsigned i=0; i<upNames.size();i++){
    cout<<"#"<<i<<":\t"<<upNames[i]<<endl;
    for(unsigned j=0; j<cids.size(); j++){
      if(cids[j]==cid)continue;
      SSIDL::array1<string> cps=bs->getCompatiblePortList(cid,upNames[i],cids[j]);
      for(unsigned k=0; k<cps.size(); k++){
	cout<<"\t|___ "<<cps[k]<<"\t("<<cids[j]->getInstanceName()<<")"<<endl;
      }
    }
  } 
}

void TxtBuilder::list_connections(string args[])
{
  vector<ComponentID::pointer> cids;
  cids.push_back(bs->getComponentID(args[0]));
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);

  cout<<"Connections of component "<<args[0]<<":\n";
  for(unsigned i=0; i<connIDs.size();i++){
    if(connIDs[i]->getProvider()==cids[0])
      cout<<"#"<<i<<":\t"<<connIDs[i]->getProviderPortName()<<"<--"<<connIDs[i]->getUserPortName()<<" ("<<connIDs[i]->getUser()->getInstanceName()<<")\n";
    else
      cout<<"#"<<i<<":\t"<<connIDs[i]->getUserPortName()<<"-->"<<connIDs[i]->getProviderPortName()<<" ("<<connIDs[i]->getProvider()->getInstanceName()<<")\n";
  }
}

void TxtBuilder::create(string args[])
{
  bs->createInstance(args[0],args[0],TypeMap::pointer(0));
}

void TxtBuilder::connect(string args[])
{
  ComponentID::pointer cid1=bs->getComponentID(args[0]);
  string &p1=args[1];
  ComponentID::pointer cid2=bs->getComponentID(args[2]);
  string &p2=args[3];
  bs->connect(cid1,p1,cid2,p2);
}

void TxtBuilder::go(string args[])
{
  ComponentID::pointer cid=bs->getComponentID(args[0]);
  bool hasGoPort=false;
  bool isSciPort=false;
  SSIDL::array1<string> ports = bs->getProvidedPortNames(cid);
  for(unsigned int i=0; i < ports.size(); i++){
    if(ports[i]=="go") hasGoPort=true;
    if(ports[i]=="sci.go"){
      hasGoPort=true;
      isSciPort=true;
    }
  }

  if(hasGoPort)
  {
    string instanceName = cid->getInstanceName();
    string goPortName = instanceName+" goPort";
    svc->registerUsesPort(goPortName, "sci.cca.ports.GoPort",
			  sci::cca::TypeMap::pointer(0));
    ConnectionID::pointer connID=bs->connect(svc->getComponentID(), goPortName, cid, isSciPort?"sci.go":"go");
    Port::pointer p = svc->getPort(goPortName);
    ports::GoPort::pointer goPort = pidl_cast<ports::GoPort::pointer>(p);
    if(goPort.isNull()){
      cerr << "goPort is not connected, cannot bring up Go!\n";
    } 
    else{
      int status=goPort->go();
      cout<<"Done! return value="<<status<<endl;
    }
    svc->releasePort(goPortName);
    bs->disconnect(connID,0);
    svc->unregisterUsesPort(goPortName);
  }
}

void TxtBuilder::disconnect(string args[])
{
  vector<ComponentID::pointer> cids;
  cids.push_back(bs->getComponentID(args[0]));
  string port=args[1];
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
  for(unsigned i=0; i<connIDs.size();i++){
    if( (connIDs[i]->getProvider()==cids[0] && port==connIDs[i]->getProviderPortName())
	||(connIDs[i]->getUser()==cids[0] && port==connIDs[i]->getUserPortName())){
      cout<<"\t"<<connIDs[i]->getUserPortName()
	  <<"("<<connIDs[i]->getUser()->getInstanceName() <<")"
	  <<"-X->"<<connIDs[i]->getProviderPortName()
	  <<" ("<<connIDs[i]->getProvider()->getInstanceName()<<")\n";
      bs->disconnect(connIDs[i],0);
      
    }
  }
}

void TxtBuilder::destroy(string args[])
{
  vector<ComponentID::pointer> cids;
  cids.push_back(bs->getComponentID(args[0]));
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
  for(unsigned i=0; i<connIDs.size();i++){
    bs->disconnect(connIDs[i],0);
  }
  bs->destroyInstance(cids[0],0);
}

void TxtBuilder::execute(string args[])
{
  ifstream file(args[0].c_str());
  if(!file.is_open()){
    cerr<<"Fail to open file "<<args[0]<<endl;
    return;
  }
  while(!file.eof()){
    const int n=256;
    char cmdline[n];
    file.getline(cmdline,n);
    if(string(cmdline).find("EOF",0)==0) break;
    cout<<endl<<"Executing: "<<cmdline<<endl;
    exec_command(cmdline);
  }
}

void TxtBuilder::load(string args[])
{
  cerr<<"args="<<args[0]<<"$"<<args[1]<<"$"<<args[2]<<"$"<<args[3]<<"$"<<endl;
}

void TxtBuilder::save_as(string args[])
{
  cerr<<"args="<<args[0]<<"$"<<args[1]<<"$"<<args[2]<<"$"<<args[3]<<"$"<<endl;
}

void TxtBuilder::save(string args[])
{
  cerr<<"args="<<args[0]<<"$"<<args[1]<<"$"<<args[2]<<"$"<<args[3]<<"$"<<endl;
}

void TxtBuilder::insert(string args[])
{
  cerr<<"args="<<args[0]<<"$"<<args[1]<<"$"<<args[2]<<"$"<<args[3]<<"$"<<endl;
}

void TxtBuilder::clear(string args[])
{
  cerr<<"args="<<args[0]<<"$"<<args[1]<<"$"<<args[2]<<"$"<<args[3]<<"$"<<endl;
}

void TxtBuilder::quit(string args[])
{
  svc->releasePort("cca.BuilderService"); 
  cout<<"TxtBuilder quits!\n";
  exit(0);
}

void TxtBuilder::help(string args[])
{
  int num_cmds=sizeof(cmddesc)/sizeof(char *);
  for(int i=0; i<num_cmds;i++){
    cout<<cmddesc[i]<<endl;
  }
}

void TxtBuilder::bad_command(string args[])
{
  cout<<"Bad command!\n";
}


/*
void myBuilderPort::buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
				    const std::string &frameworkURL)
{
  builder->buildRemotePackageMenus(reg, frameworkURL);
}
*/




