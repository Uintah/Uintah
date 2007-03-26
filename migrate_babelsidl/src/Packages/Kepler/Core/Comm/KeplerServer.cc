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

//prototype ptolemy server
//by oscar barney

#include <Packages/Kepler/Core/Comm/KeplerServer.h>
#include <Packages/Kepler/Core/Comm/NetworkHelper.h>
#include <Packages/Kepler/Core/Util/CStringProcessor.h>

#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TCLTask.h>

#include <Core/Util/Timer.h>
//#include <Core/Util/Socket.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/NetworkIO.h>
#include <Dataflow/Network/Scheduler.h>

#include <Dataflow/Modules/Render/Viewer.h>

#include <sstream>

class ServerTime : public Runnable
{
public:
  ServerTime(double max) : maxTime_(max) {}
  virtual ~ServerTime() {}
  void run();
  void startTimer() { throttle.start(); }
  void stopTimer() { throttle.stop(); }
  void clearTimer() { throttle.clear(); }
  void stopAndClearTimer()
  {
    throttle.stop();
    throttle.clear();
  }

private:
  double maxTime_;  //TODO right now this is hard coded..  make command line option?
  TimeThrottle throttle;
};

void ServerTime::run()
{
  throttle.start();
  while (true) {
    std::cerr << "ServerTime::run()" << std::endl;
    throttle.wait_for_time(maxTime_);
    //std::string state = (throttle.current_state() == Timer::Running) ? "Running" : "Stopped";
    //std::cerr << "ServerTime::run(): wait done, throttle state=" << state << std::endl;
    if (throttle.current_state() == Timer::Running) {
      GuiInterface *gui = GuiInterface::getSingleton();
      ASSERT(gui);
      gui->eval("exit");
    } else { // throttle state == Timer::Stopped
      Thread::yield();
    }
  }
}

KeplerServer::KeplerServer(Network *n, const double timeout) :  MAX_TIME_SECONDS(timeout), THREAD_STACK_SIZE(1024*2), net(n)
{
  {
    Mutex m("lock workerCount init");
    Guard g(&m);
    if (! workerCount) {
      workerCount = new AtomicCounter("worker count", 0);
    }
  }
}

//TODO figure out if we want to close this and where/when
//Close(listenfd);  no closed when we quit but what if two
//people log into the same machine and both want to use
//SCIRun then in this case they will have to get different ports
// or somehow share the same port.......

KeplerServer::~KeplerServer()
{
  {
    Mutex m("lock workerCount delete");
    Guard g(&m);
    if (workerCount) {
      delete workerCount;
    }
  }
  Close(listenfd);
  //cout << "Listen closed" << endl;
}

void KeplerServer::run()
{
  servSem().down();   //TODO may not need but could
  //be useful when we get to the point where pt starts up
  //the server bc we wait for main to finish
  ProcessRequest::processRequestSem().up();//disallows simultanious work requests

  //networking variables
  int connfd;
  socklen_t  clientLength;
  struct sockaddr_in clientAddr, serverAddr;

  // start up the server
  startUpServer(&listenfd, &serverAddr);
  // set up a timeout thread
  ServerTime *st = new ServerTime(MAX_TIME_SECONDS);
  Thread *idleTime = new Thread(st, "time idleness", 0, Thread::NotActivated);
  idleTime->setDaemon();
  idleTime->setStackSize(THREAD_STACK_SIZE);
  idleTime->activate(false);
  idleTime->detach();

  servSem().up();

  while (true) {
    std::cerr << "KeplerServer looping!" << std::endl;
    clientLength = sizeof(clientAddr);
//std::cerr << "KeplerServer: try accept" << std::endl;
    connfd = Accept(listenfd, (SA *) &clientAddr, &clientLength);
//std::cerr << "KeplerServer: accepted" << std::endl;

    servSem().down();
    ++(*KeplerServer::workerCount);
    if (*KeplerServer::workerCount == 1 && KeplerServer::state != KeplerServer::Detached) {
      KeplerServer::state = KeplerServer::Iterating;  //set state to running
      //stop the timer thread
      idleTime->stop();
      ((ServerTime*) idleTime->getRunnable())->stopAndClearTimer();
    } //TODO fatal errors after this need to start time out timer back up
    servSem().up();

    ProcessRequest *proc_req = new ProcessRequest(net, connfd, idleTime);
    Thread *pr = new Thread(proc_req, "process client request", 0, Thread::NotActivated);
    pr->setDaemon();
    pr->setStackSize(1024*512);
    pr->activate(false);
    pr->detach();
  }
}

Semaphore& KeplerServer::servSem()
{
  static Semaphore sem_("pt server semaphore", 0);
  return sem_;
}

ProcessRequest::~ProcessRequest()
{
  Close(connfd);
}

void ProcessRequest::run()
{
  cout << "Here and listening" << endl;

  //TODO based on latest changes we have to get the gui here instead of in main.cc...
  //need to remove that and change the constructors!
  gui = GuiInterface::getSingleton();
  ASSERT(gui);

  //networking variables
  ssize_t n;
  char line[MAXLINE];   //lines we get sent
  std::string task = "";

  //read in task
  if( (n = Readline(connfd, line, MAXLINE)) == 0){
    print_error("connection closed by other end");
  }
  task = std::string(line);
  task = task.substr(0, task.size()-1);
  cout << "task : " << task << endl;

  // process the request
  if (task == "iterate") {
    processRequestSem().down();
    processItrRequest(connfd);
    processRequestSem().up();
  } else if (task == "quit") {
    quit(connfd);
  } else if (task == "stop") {
    stop(connfd);
  } else if(task == "detach") {
    detach(connfd);
  } else if(task == "eval") {
    if ( (n = Readline(connfd, line, MAXLINE)) == 0) {
      print_error("connection closed by other end");
    }
    eval(connfd, std::string(line));
  } else {
    //TODO handle this because it could be a hacker
    std::cerr << "unknown task: " << task << std::endl;
    abort();
  }

  KeplerServer::servSem().down();
  --(*KeplerServer::workerCount);
  //if we are the last thread start timer and set state to idle
  if (*KeplerServer::workerCount == 0 && KeplerServer::state != KeplerServer::Detached) {
    ((ServerTime*) idleTime->getRunnable())->startTimer();
    idleTime->resume();
    KeplerServer::state = KeplerServer::Listening;
  }
  KeplerServer::servSem().up();
}

Semaphore& ProcessRequest::iterSem()
{
  static Semaphore sem_("iterate semaphore", 0);
  return sem_;
}

Semaphore& ProcessRequest::processRequestSem()
{
  static Semaphore sem_("process request semaphore", 0);
  return sem_;
}

//Note that if you add this callback at the beginning of a task
//it is necessary to remove it at the end so it wont modify the static
//semephore in the future when it is needed again.
bool ProcessRequest::iter_callback(void *data)
{
  iterSem().up();
  return false;
}

std::string ProcessRequest::Iterate(std::vector<std::string> doOnce, int size1, std::vector<std::string> iterate, int size2, int numParams, std::string picPath, std::string picFormat)
{
  std::string name;
  std::string imageName;

  //get a pointer to the viewer if we need it and check to see if its valid
  Viewer* viewer;
  if (! picPath.empty()){
    viewer = (Viewer*) net->get_module_by_id("SCIRun_Render_Viewer_0");
    if (viewer == 0) {
      //returnValue = "no viewer present";
      return "no viewer present";
    }
  }

  Scheduler* sched = net->get_scheduler();
  sched->add_callback(iter_callback, NULL);

  //set the initial parameters
  Module* modptr;
  GuiInterface* modGui;

  for(int i = 0; i < size1; i++){
    modptr = net->get_module_by_id(doOnce[i]);
    if(modptr == 0){
      //returnValue = doOnce[i] + " not present in the network";
      sched->remove_callback(iter_callback, 0);
      return doOnce[i] + " not present in the network";
    }
    i++;
    modGui = modptr->get_gui();
    ASSERT(modGui);

    //std::cout << "doOnce " << doOnce[i-1] << " " << doOnce[i] << " " << doOnce[i+1] << std::endl;
    modGui->set("::" + doOnce[i-1] + doOnce[i], doOnce[i+1]);
    // kludge:
    imageName = basename(doOnce[i+1]);
    i++;
  }


  //iterate through the tasks given to SCIRun
  for (int i = 0; i < numParams; i++){
    if (KeplerServer::state == KeplerServer::Listening) {
      sched->remove_callback(iter_callback, 0);
      return "early abort";
    }

    for (int j = 0; j < size2; j = j + numParams - i) {
      //TODO ask if it would be better here to have a dynamically
      //allocated array of module pointers for each thing
      //depends on how efficient getmodbyid really is
      modptr = net->get_module_by_id(iterate[j]);
      if (modptr == 0) {
        //returnValue = iterate[j] + " not present in the network";
        sched->remove_callback(iter_callback, 0);
        return iterate[j] + " not present in the network";
      }
      j++;
      modGui = modptr->get_gui();
      ASSERT(modGui);

      //std::cout << "iterate " << iterate[j-1] << " " << iterate[j] << " " << iterate[j+i+1] << std::endl;

      modGui->set("::" + iterate[j-1] + iterate[j], iterate[j+i+1]);
      j = j + i + 1;
    }

    //execute all and wait for it to finish
    gui->eval("updateRunDateAndTime {0}");
    gui->eval("netedit scheduleall");
    iterSem().down();

    //if you want to save the picture
    //TODO do we care if there is no viewer that is getting the messages?
    //TODO worry about saving over existing images.  would be cool to prompt
    // the user if they are going to save over an image that exists already
    if (! picPath.empty()) {
      //std::cerr << "Save pictures!" << std::endl;
      int p = imageName.rfind(".");
      imageName.erase(p);
      name = picPath + imageName + to_string(i) + "" + picFormat;
std::cerr << "Save pictures: " << name << std::endl;

      //when the viewer is done save the image
      // old size: 640x470

      gui->eval("SciRaise .uiSCIRun_Render_Viewer_0-ViewWindow_0");
      gui->eval("focus .uiSCIRun_Render_Viewer_0-ViewWindow_0");

      // should picFormat have leading '.'?
      ViewerMessage *msg1 = scinew ViewerMessage(MessageTypes::ViewWindowDumpImage,"::SCIRun_Render_Viewer_0-ViewWindow_0", name, "by_extension", "640","479");
      viewer->mailbox_.send(msg1);

      ViewerMessage *msg2 = scinew ViewerMessage("::SCIRun_Render_Viewer_0-ViewWindow_0");
      viewer->mailbox_.send(msg2);

    } //else we do not try and save pictures

  }

  sched->remove_callback(iter_callback, 0);
  return "OK";
}

void ProcessRequest::processItrRequest(int sockfd)
{
  ssize_t n;   //return value of some network functions
  char line[MAXLINE];   //lines we get sent
  char *retVal;   //value we will return to client
  std::string rv("none");    //return value for a run

  std::string netFile, picPath, picFormat, temp; //path were pictures will be saved
  int size1, size2, numParams;   //size of each input, number of iterations
  std::vector<std::string> input1;
  std::vector<std::string> input2;


  //read in string with initial settings
  if ( (n = Readline(sockfd, line, MAXLINE)) == 0) {
    print_error("connection closed by peer");
  }
  //now put all the stuff in array
  processCString(line, input1, size1);

  //read in string that has values to iterate over
  if ( (n = Readline(sockfd, line, MAXLINE)) == 0) {
    print_error("connection closed by peer");
  }
  //now put all stuff in array
  processCString(line, input2, size2);

  //read in the number of parameters
  if ( (n = Readline(sockfd, line, MAXLINE)) == 0) {
    print_error("connection closed by peer");
  }
  numParams = atoi(line);

  //get network file path/name
  if ( (n = Readline(sockfd, line, MAXLINE)) == 0) {
    print_error("connection closed by peer");
  }
  netFile = std::string(line);
  netFile = netFile.substr(0,netFile.size() - 1);
  ASSERT(gui);

  if (KeplerServer::loadedNet != netFile) { // TODO: useless comparison???
    temp = gui->eval("ClearCanvas 0");  //clear the net
    //temp seems to be 0 always
    //TODO this yield is probably due to scirun error that needs to be fixed
    //Thread::yield();  //necessary to avoid a "Error: bad window path name"
    std::cout << "loaded net " << KeplerServer::loadedNet << "!" << netFile << std::endl;
    if (ends_with(netFile, ".net")) {
      temp = gui->eval("source " + netFile);
    } else {
      // attempt to load the file and let NetworkIO handle the consequences
      NetworkIO::load_net(netFile);
      NetworkIO ln;
      ln.load_network();
    }
    KeplerServer::loadedNet = netFile;
  }

  //read in picPath
  if( (n = Readline(sockfd, line, MAXLINE)) == 0){
    print_error("connection closed by peer");
  }
  picPath = std::string(line);
  //std::cerr << "picPath=" << picPath << std::endl;
  picPath = picPath.substr(0,picPath.size()-1);

  //read in the imageFormat
  if( (n = Readline(sockfd, line, MAXLINE)) == 0){
    print_error("connection closed by peer");
  }
  picFormat = std::string(line);
  picFormat = picFormat.substr(0,picFormat.size()-1);

  rv = Iterate(input1, size1, input2, size2, numParams, picPath, picFormat);
  cout << "done iterating rv = " << rv << "??" << endl;

  retVal = ccast_unsafe(rv);
  Written(sockfd, retVal, rv.size());
}

void ProcessRequest::quit(int sockfd)
{
  char message[10] = "quitting\n";
  Written(sockfd, message, 9);
  ASSERT(gui);

  gui->eval("exit");
}

void ProcessRequest::stop(int sockfd)
{
  char message[9] = "stopped\n";
  Written(sockfd, message, 8);
  KeplerServer::servSem().down();
  //TODO may need to fix this in relation to detached
  KeplerServer::state = KeplerServer::Listening;

  KeplerServer::servSem().up();
}

void ProcessRequest::detach(int sockfd)
{
  std::stringstream ss;
  ss << KeplerServer::nextTag;
  std::string temp = ss.str();
  //cout << "temp: " << temp << "len: " << temp.length() << endl;
  char message[10] = "detached\n";
  Written(sockfd, message, 9);
  Written(sockfd, (void*)temp.c_str(),temp.length());
  KeplerServer::servSem().down();
  KeplerServer::state = KeplerServer::Detached;
  KeplerServer::servSem().up();
}

void ProcessRequest::eval(int sockfd, std::string command)
{
  std::string temp;
  int retVal = gui->eval(command, temp);
  if (retVal != 1)
    std::cerr << "gui->eval error" << std::endl;
  Written(sockfd, (void*) temp.c_str(), temp.length());
}
