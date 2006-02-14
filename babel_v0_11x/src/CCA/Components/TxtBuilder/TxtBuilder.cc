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


/*
 *  TxtBuilder.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#include <CCA/Components/TxtBuilder/TxtBuilder.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/CCA/CCAException.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <stdlib.h>
#include <curses.h>
#include <panel.h>
#include <menu.h>
#include <stdlib.h>
#include <iostream>
#include <CCA/Components/TxtBuilder/DropMenu.h>
#include <CCA/Components/TxtBuilder/TxtModule.h>
#include <CCA/Components/TxtBuilder/TxtMenu.h>
#include <CCA/Components/TxtBuilder/TxtNetwork.h>
#include <CCA/Components/TxtBuilder/TxtConnection.h>
#include <CCA/Components/TxtBuilder/TxtMessage.h>
#include <SCIRun/TypeMap.h>

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

TxtMessage *
TxtBuilder::tm=NULL;

sci::cca::Services::pointer
TxtBuilder::svc(0);

sci::cca::ports::BuilderService::pointer
TxtBuilder::bs(0);

WINDOW*
TxtBuilder::win_cmd;

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
  if(cr.isNull()){
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
  "create <comp>:               create component <comp>",		
  "connect <c1> <p1> <c2> <p2>: connect port <p1> (of component <c1>) and port <p2> (of component <c2>)",
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


  PANEL  *pan_menu;
  PANEL  *pan_main;
  PANEL  *pan_msg;
  PANEL  *pan_cmd;

  initscr();
  keypad(stdscr, TRUE);
  cbreak();
  noecho();

  int lines_menu=1;
  int cols_menu=COLS;


  int lines_cmd=1;
  int cols_cmd=COLS;


  int lines_msg=10;
  int cols_msg=COLS;

  int lines_main=LINES-lines_menu-lines_msg-lines_cmd;
  int cols_main=COLS;

  int y_menu=0;
  int y_main=y_menu+lines_menu;
  int y_msg=y_main+lines_main;
  int y_cmd=y_msg+lines_msg;
  


  /* Create windows for the panels */
  win_menu = newwin(lines_menu, cols_menu, y_menu, 0);
  win_main = newwin(lines_main, cols_main, y_main, 0);
  win_msg  = newwin(lines_msg, cols_msg, y_msg, 0);
  win_cmd  = newwin(lines_cmd, cols_cmd, y_cmd, 0);
  WINDOW *swin_msg = derwin(win_msg,lines_msg-2, cols_msg-2, 1, 1);
  tm=new TxtMessage(swin_msg);
  
  // Create borders around the windows so that you can see the effect of panels
  box(win_main, 0, 0);
  box(win_msg,  0, 0);

  //  cerr<<"y="<<y_menu<<" "<<y_main<<" "<<y_msg<<" "<<y_cmd<<endl;

  //Attach a panel to each window, order: newest on top
  pan_menu = new_panel(win_menu); 	
  pan_main = new_panel(win_main);
  pan_msg = new_panel(win_msg);
  pan_cmd = new_panel(win_cmd);

  TxtMenu menu(svc);
  menu.setup(win_menu);
  network.setup(win_main);
  mvwprintw(win_cmd,0,0,"F9 - Menu  F5 - Network Window  : - Command Window  r - Refresh Windows");
  // Update the stacking order
  update_panels();

  // Show it on the screen 
  doupdate();

  displayMessage("SCIRun2 Started");

  //the main event loop
  int c;
  while(true){
    switch(c=wgetch(stdscr)){	
    case KEY_F(9):
      menu.enter(win_menu);
      break;
    case KEY_F(5):
      network.enter();
      break;
    case KEY_UP:
      tm->print(-(lines_msg-2-1)*2, (lines_msg-2) );
      break;
    case KEY_DOWN: 
      tm->print(  0, lines_msg-2);
      break;
    case ':': 
      {
	string cmdline=getString(":");
	exec_command(cmdline.c_str());
      }
      break;
    case 'r': 
      TxtNetwork::drawConnections();
      update_panels();
      doupdate();
      break;
    default:
      //Command::process
      //printf("you just typed 0x%x\n",c);
      break;
    }
  }

  /////////////////////////
  refresh();
  endwin();

  /*
  while(true){
    const int n=256;
    char cmdline[n];
    cout<<"\nSCIRun>";
    cin.getline(cmdline,n);
    if(exec_command(cmdline))break;
  }
  */
}

bool TxtBuilder::exec_command(const char* cmdline)
{
   string args[4];
   int cmd=parse(cmdline,args);
   if (cmd==USE_CLUSTER) use_cluster(args);
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
   else if(cmd==QUIT) quit(args);
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

void TxtBuilder::create(string args[])
{
  sci::cca::TypeMap *tm = new SCIRun::TypeMap;
  //  tm->putString("LOADER NAME", "");
  //  tm->putString("cca.className", "CCA"); // component type
  //  displayMessage(args[0]);
  sci::cca::ComponentID::pointer cid=bs->createInstance(args[0],args[0],TypeMap::pointer(tm));
  TxtNetwork::addModule(new TxtModule(cid,args[0]));

}

void TxtBuilder::connect(string args[])
{
  ComponentID::pointer cid1=bs->getComponentID(args[0]);
  string &p1=args[1];
  ComponentID::pointer cid2=bs->getComponentID(args[2]);
  string &p2=args[3];
  bs->connect(cid1,p1,cid2,p2);
  TxtNetwork::drawConnections();
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
      displayMessage("goPort is not connected, cannot bring up Go");
    } 
    else{
      /*int status=*/goPort->go();
      //cout<<"Done! return value="<<status<<endl;
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
      bs->disconnect(connIDs[i],0);
      break;
    }
  }
  TxtNetwork::drawConnections();
}

void TxtBuilder::destroy(string args[])
{
  vector<ComponentID::pointer> cids;
  cids.push_back(bs->getComponentID(args[0]));
  SSIDL::array1<ConnectionID::pointer> connIDs=bs->getConnectionIDs(cids);
  for(unsigned i=0; i<connIDs.size();i++){
    bs->disconnect(connIDs[i],0);
  }
  TxtNetwork::delModule(cids[0]);
  bs->destroyInstance(cids[0],0);
}

void TxtBuilder::execute(string args[])
{
  ifstream file(args[0].c_str());
  if(!file.is_open()){
    displayMessage("Fail to open file ");//<<args[0]<<endl;
    return;
  }
  while(!file.eof()){
    const int n=256;
    char cmdline[n];
    file.getline(cmdline,n);
    if(string(cmdline).find("EOF",0)==0) break;
    displayMessage("Executing: ");//<<cmdline<<endl;
    exec_command(cmdline);
  }
}

void TxtBuilder::load(string args[])
{
  TxtNetwork::setFilename(args[0]);
  TxtNetwork::loadFile();
}

void TxtBuilder::save_as(string args[])
{
  TxtNetwork::setFilename(args[0]);
  if (TxtNetwork::getFilename().size()>0) {
    TxtNetwork::writeFile();
  }
}

void TxtBuilder::save(string args[])
{
  if (TxtNetwork::getFilename().size()==0) {
    TxtNetwork::setFilename(SCIRun::TxtBuilder::getString("Save As:"));
  }
  if (TxtNetwork::getFilename().size()>0) {
    TxtNetwork::writeFile();
  } 
}

void TxtBuilder::insert(string args[])
{
  displayMessage("insert <network> is not implemented!");
}

void TxtBuilder::clear(string args[])
{
  TxtNetwork::clear();
}

void TxtBuilder::quit(string args[])
{
  svc->releasePort("cca.BuilderService"); 
  Thread::exitAll(0);
}

void TxtBuilder::help(string args[])
{
  int num_cmds=sizeof(cmddesc)/sizeof(char *);
  for(int i=0; i<num_cmds;i++){
    displayMessage(cmddesc[i]);
  }
}

void TxtBuilder::bad_command(string args[])
{
  displayMessage("Bad command!");
}

sci::cca::Services::pointer 
TxtBuilder::getService(){
  return svc;
}

sci::cca::ports::BuilderService::pointer 
TxtBuilder::getBuilderService(){
  return bs;
}

/*
void myBuilderPort::buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
				    const std::string &frameworkURL)
{
  builder->buildRemotePackageMenus(reg, frameworkURL);
}
*/

void
TxtBuilder::displayMessage(const string &msg){
  tm->add(msg);
}

string 
TxtBuilder::getString(string prompt){
  string s=prompt;
  int c;
  werase(win_cmd);
  mvwprintw(win_cmd,0,0,s.c_str());
  wrefresh(win_cmd);
  while(true){
    switch(c=wgetch(stdscr)){	
    case 0x107: //BACKSPACE
      if(s.size()>prompt.size()){
	werase(win_cmd);
	s=s.substr(0, s.size()-1);
	mvwprintw(win_cmd,0,0,s.c_str());
	wrefresh(win_cmd);
      }
      break;
    case 0xa: //ENTER
      werase(win_cmd);
      wrefresh(win_cmd);
      mvwprintw(win_cmd,0,0,"F9 - Menu  F5 - Network Window  : - Command Window  r - Refresh Windows");
      return s.substr(prompt.size(), s.size()-prompt.size());
    case 0x1b: //ESC
      werase(win_cmd);
      wrefresh(win_cmd);
      mvwprintw(win_cmd,0,0,"F9 - Menu  F5 - Network Window  : - Command Window  r - Refresh Windows");
      return "";
    default:
      s=s+(char)c;
      werase(win_cmd);
      mvwprintw(win_cmd, 0,0, s.c_str());
      wrefresh(win_cmd);
      break;
    }
  }
}
