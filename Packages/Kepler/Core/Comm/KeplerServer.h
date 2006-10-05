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


#ifndef Kepler_Core_Comm_KeplerServer_h
#define Kepler_Core_Comm_KeplerServer_h

#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Guard.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/AtomicCounter.h>
#include <Dataflow/Network/Network.h>

using namespace SCIRun;

//class Socket;

class KeplerServer : public Runnable
{
public:
  enum State {
    Listening = 0,
    Iterating,
    Detached,
  };

  KeplerServer(Network *n);
  ~KeplerServer();
  void run();
  static Semaphore& servSem();
  static std::string loadedNet;
  static State state;
  static int nextTag; // this does what???
  static AtomicCounter* workerCount;

  const double MAX_TIME_SECONDS;
  const int THREAD_STACK_SIZE;

  //std::map<int, string> saved_results; //TODO make static?
  //store stuff in it etc.... stoped this thought here.
  //so detach just stops the timer right now

private:
  Network *net;
  int listenfd;
};

std::string KeplerServer::loadedNet;
AtomicCounter* KeplerServer::workerCount = 0;
KeplerServer::State KeplerServer::state = KeplerServer::Listening;
int KeplerServer::nextTag = 0;

class ProcessRequest : public Runnable
{
public:
  ProcessRequest(Network *n, int fd, Thread* t)
    : gui(0), net(n), connfd(fd), idleTime(t) {}
  ~ProcessRequest();
  void run();
  static Semaphore& processRequestSem();    //disallows two iterate requests at once
  void processItrRequest(int sockfd);

private:
  static Semaphore& iterSem();
  static bool iter_callback(void *data);
  string Iterate(vector<string> doOnce, int size1, vector<string> iterate, int size2, int numParams, string picPath, string picFormat);
  void quit(int sockfd);
  void stop(int sockfd);
  void detach(int sockfd);
  void eval(int sockfd, string command);

  GuiInterface *gui;
  Network *net;
  string loadedNet;
  int connfd;
  Thread *idleTime;
};

#endif
