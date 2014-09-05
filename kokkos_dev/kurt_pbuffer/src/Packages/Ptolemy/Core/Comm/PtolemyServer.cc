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

#include	<Packages/Ptolemy/Core/Comm/PtolemyServer.h>
#include	<Packages/Ptolemy/Core/Comm/NetworkHelper.h>
#include <Packages/Ptolemy/Core/Comm/PTstring.h>

#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Thread/Thread.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Modules/Render/Viewer.h>
#include <Dataflow/Network/Scheduler.h>



void ServerTime::run()
{
	double temp = 0.0;
	while(true){
		temp =wc->time();
		//cout << "time: " << temp << endl;
		if(temp>max_time){
			cout << "timed out " << endl;
			gui->eval("exit");
		}
		else{
			Thread::yield();
		}
	}
}		

	//TODO figure out if we want to close this and where/when
	//Close(listenfd);  no closed when we quit but what if two
	//people log into the same machine and both want to use
	//SCIRun then in this case they will have to get different ports
	// or somehow share the same port.......       so we have a security
	// issue in that people that both want to use the same machine
	//may be talking to the wrong person's SCIRun.		

PtolemyServer::~PtolemyServer()
{
	Close(listenfd);
	//cout << "LIsten closed" << endl;
}	
	
void PtolemyServer::run()
{
	servSem().down();		//TODO may not need but could
	//be useful when we get to the point where pt starts up 
	//the server bc we wait for main to finish
	
	//networking variables
	int connfd;
	socklen_t	 clientLength;
	struct sockaddr_in clientAddr, serverAddr;
	
	//start up the server
	startUpServer(&listenfd, &serverAddr);
	//time how long the server has been running
	WallClockTimer *wc = new WallClockTimer();
	wc->start();
	ServerTime *st = new ServerTime(gui, wc, 60.0);
	Thread *idle_time = new Thread(st, "time idleness",0, Thread::NotActivated);
	idle_time->setDaemon();
	idle_time->setStackSize(1024*2);
	idle_time->activate(false);
	idle_time->detach();
	
	servSem().up();
	
	while(true){
		clientLength = sizeof(clientAddr);
		connfd = Accept(listenfd, (SA *) &clientAddr, &clientLength);
		cout << "here stopping timer and accepting" << endl;
		
		servSem().down();
			PtolemyServer::state = 1;  //set state to running
			PtolemyServer::worker_count++;
			if(PtolemyServer::worker_count==1){
				//stop the timer thread
				idle_time->stop();
				wc->stop();
				wc->clear();  //clear time because something happened
			}
		servSem().up();
		
		ProcessRequest *proc_req = new ProcessRequest(gui,net,connfd,idle_time,wc);
		Thread *pr = new Thread(proc_req,"process client request", 0, Thread::NotActivated);
		pr->setDaemon();
		pr->setStackSize(1024*512);
		pr->activate(false);
		pr->detach();
	}
}

Semaphore& PtolemyServer::servSem()
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
	//iterSem().down();  //right now it starts down so we dont have to do it	
	cout << "Here and listening" << endl;
	
	//networking variables
	ssize_t n;
	char line[MAXLINE];	  //lines we get sent  
	string task = "";
	
	//read in task
	if( (n = Readline(connfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	task = string(line);
	task = task.substr(0, task.size()-1);
	cout << "task : " << task << endl;

		// process the request 
	if(task == "iterate"){
		processItrRequest(connfd);
	}else if (task == "quit"){
		quit(connfd);
	}
	else if (task == "stop"){
		stop(connfd);
	}
	

	PtolemyServer::servSem().down();
	PtolemyServer::worker_count--;
	//if we are the last thread start timer and set state to idle
	if(PtolemyServer::worker_count==0){
		wc->start();
		idle_time->resume(); 
		PtolemyServer::state = 0;
	}
	PtolemyServer::servSem().up();
}

Semaphore& ProcessRequest::iterSem()
{
	static Semaphore sem_("iterate semaphore", 0);
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



string ProcessRequest::Iterate(vector<string> doOnce, int size1, vector<string> iterate, int size2, int numParams, string picPath, string picFormat)
{
	//iterSem().down();  //we block sooner now... as soon as a request is made
	string name;

	//get a pointer to the viewer if we need it and check to see if its valid
	Viewer* viewer;
	if(picPath != ""){
		viewer = (Viewer*)net->get_module_by_id("SCIRun_Render_Viewer_0");
		if(viewer == 0){
			//returnValue = "no viewer present";
			iterSem().up();
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
			iterSem().up();
			return doOnce[i] + " not present in the network";
		}
		i++;
		modGui = modptr->getGui();
		ASSERT(modGui);
		
		//std::cout << "doOnce " << doOnce[i-1] << " " << doOnce[i] << " " << doOnce[i+1] << std::endl;
		modGui->set("::" + doOnce[i-1] + doOnce[i], doOnce[i+1]);
		i++;
	}
	
	
	//iterate through the tasks given to SCIRun
	for(int i = 0; i < numParams; i++){
		if (PtolemyServer::state == 0){
			sched->remove_callback(iter_callback, 0);
			iterSem().up();
			return "early abort";
		}
		for(int j = 0; j < size2; j=j+numParams-i){
			//TODO ask if it would be better here to have a dynamically
			//allocated array of module pointers for each thing
			//depends on how efficient getmodbyid really is
			modptr = net->get_module_by_id(iterate[j]);
			if(modptr == 0){
				//returnValue = iterate[j] + " not present in the network";
				sched->remove_callback(iter_callback, 0);
				iterSem().up();
				return iterate[j] + " not present in the network";
			}
			j++;
			modGui = modptr->getGui();
			ASSERT(modGui);
		
			//std::cout << "iterate " << iterate[j-1] << " " << iterate[j] << " " << iterate[j+i+1] << std::endl;
			
			modGui->set("::" + iterate[j-1] + iterate[j], iterate[j+i+1]);
			j=j+i+1;
		}
		
		//execute all and wait for it to finish
		gui->eval("updateRunDateAndTime {0}");
		gui->eval("netedit scheduleall");		
		iterSem().down();
		
		//if you want to save the picture
		//TODO do we care if there is no viewer that is getting the messages?
		//TODO worry about saving over existing images.  would be cool to prompt
		// the user if they are going to save over an image that exists already
		if(picPath != ""){
			name = picPath + "image" + to_string(i) + "" + picFormat;

			//when the viewer is done save the image
			ViewerMessage *msg1 = scinew ViewerMessage
					(MessageTypes::ViewWindowDumpImage,"::SCIRun_Render_Viewer_0-ViewWindow_0",name, picFormat,"640","473");
			viewer->mailbox.send(msg1); 

			ViewerMessage *msg2 = scinew ViewerMessage("::SCIRun_Render_Viewer_0-ViewWindow_0");
			viewer->mailbox.send(msg2);
		}//else we do not try and save pictures
		
	}
	
	sched->remove_callback(iter_callback, 0);
	iterSem().up();	
	return "OK";
}

void ProcessRequest::processItrRequest(int sockfd)
{
	ssize_t n;   //return value of some network functions
	char line[MAXLINE];	  //lines we get sent
	char *retVal;	  //value we will return to client
	string rv = "none";	   //return value for a run
	
	string net, picPath, picFormat, temp;	//path were pictures will be saved
	int size1, size2, numParams;   //size of each input, number of iterations  
	vector<string> input1;
	vector<string> input2;
	
	
	//read in string with initial settings
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	//now put all the stuff in array
	input1 = processCstr(line, &size1);
	
	//read in string that has values to iterate over
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	//now put all stuff in array
	input2 = processCstr(line, &size2);
		
	//read in the number of parameters
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	numParams = atoi(line);
	
	//get network file path/name
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	net = string(line);
	net = net.substr(0,net.size()-1);
	ASSERT(gui);
	
	if(PtolemyServer::loaded_net != net){
		temp = gui->eval("ClearCanvas 0");  //clear the net	
		cout << "Clear Canv result: " << temp << "!" <<endl;
		Thread::yield();  //necessary to avoid a "Error: bad window path name"
		cout << "loaded net " << PtolemyServer::loaded_net << "!" << endl;
		temp = gui->eval("source " + net);
		cout << "source result: " << temp << "!" << endl;
		PtolemyServer::loaded_net = net;
		//TODO test string for net that doesnt exist?  source doesnt
		//return a value aparntly so that doesnt work.
	}
		  
	//read in picPath
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	picPath = string(line);
	picPath = picPath.substr(0,picPath.size()-1);
	
	//read in the imageFormat
	if( (n = Readline(sockfd, line, MAXLINE)) == 0){
		print_error("connection closed by other end");
	}
	picFormat = string(line);
	picFormat = picFormat.substr(0,picFormat.size()-1);
	
	rv = Iterate(input1, size1, input2, size2, numParams, picPath, picFormat);
	cout << "done iterating rv = " << rv << "??" << endl;
	
	retVal = ccast_unsafe(rv);
	Writen(sockfd, retVal, rv.size());
	
}

void ProcessRequest::quit(int sockfd)
{
	//iterSem().down();  we block as soon as there is a request now
	char message[8] = "quiting";
	Writen(sockfd, message, 7);	
	ASSERT(gui);
	gui->eval("exit");
}

void ProcessRequest::stop(int sockfd)
{
	char message[7] = "stoped";
	Writen(sockfd, message, 6);
	PtolemyServer::servSem().down();
	PtolemyServer::state = 0;
	PtolemyServer::servSem().up();
}
