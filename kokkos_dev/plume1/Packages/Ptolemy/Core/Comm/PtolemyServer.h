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


#ifndef Ptolemy_Core_Comm_PtolemyServer_h
#define Ptolemy_Core_Comm_PtolemyServer_h

#include <Core/Util/Timer.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Dataflow/Network/Network.h>


using namespace SCIRun;

class PtolemyServer : public Runnable 
{
	public:
		PtolemyServer(TCLInterface *tclInt, Network *n) : gui(tclInt), net(n) {}
		~PtolemyServer();
		void run();
		static Semaphore& servSem();
		static string loaded_net;
		static int state;  //0 for listening, 1 for iterating, 2 for detached
		static int worker_count;
		static int next_tag;
		map<int, string> saved_results;//TODO make static? 
												//store stuff in it etc.... stoped this thought here.
												//so detach just stops the timer right now
	private:
		TCLInterface *gui;
		Network *net;
		int listenfd;
};

string PtolemyServer::loaded_net = "";
int PtolemyServer::state = 0;
int PtolemyServer::worker_count = 0;
int PtolemyServer::next_tag = 0;

class ProcessRequest : public Runnable
{
	public:
		ProcessRequest(TCLInterface *tclInt, Network *n, int fd,Thread* t, WallClockTimer* w)
	: gui(tclInt), net(n), connfd(fd), idle_time(t), wc(w) {}
		~ProcessRequest();
		void run();
		static Semaphore& mutex();		//disallows two iterate requests at once
		void processItrRequest(int sockfd);	
	private:
		static Semaphore& iterSem();
		static bool iter_callback(void *data);	
		string Iterate(vector<string> doOnce, int size1, vector<string> iterate, int size2, int numParams, string picPath, string picFormat);
		void quit(int sockfd);
		void stop(int sockfd);
		void detach(int sockfd);
		void eval(int sockfd, string command);
		TCLInterface *gui;
		Network *net;
		string loaded_net;
		int connfd;
		Thread *idle_time;	
		WallClockTimer *wc;	
};			

class ServerTime : public Runnable
{
	public:
		ServerTime(TCLInterface *tclInt, WallClockTimer* w,  double max) : gui(tclInt), wc(w), max_time(max) {}
		virtual ~ServerTime() {}
		void run();
	private:
		TCLInterface *gui;
		WallClockTimer* wc;
		double max_time;	//TODO right now this is hard coded..  make command line option?
};

#endif
