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


//TODO try load net thread do a join so we wait until it finishes? mabye
//this would help with quit scirun

void QuitSCIRun::run()
{
    // what else for shutdown?
	//Thread::exitAll(0);
	ASSERT(gui);
	cout << "about to exit" << endl;
	gui->eval("exit");  //WORKS in main but rarely here!
}

void LoadNet::run()
{
	string netName = "/scratch/test/test2.net";
	ASSERT(gui);
	cout << "about to load" << endl;
	string evalret = gui->eval("loadnet " + netName);
	cout << "eval " << evalret << endl;
	//gui->execute("loadnet {" + netName + "}");
	//cout << "loaded" << endl;
	PtolemyServer::iterSem().up();
}

		//TODO figure out if we want to close this and where/when
		//Close(listenfd);  no closed when we quit but what if two
		//people log into the same machine and both want to use
		//SCIRun then in this case they will have to get different ports
		// or somehow share the same port.......       so we have a security
		// issue in that people that both want to use the same machine
		//may be talking to the wrong person's SCIRun.		
		
void PtolemyServer::run()
{
	//iterSem().down();
	//cout << "net should be loaded" << endl;
	
	//networking variables
	ssize_t n;
	char line[MAXLINE];	  //lines we get sent  
	int listenfd, connfd;
	socklen_t	 clientLength;
	struct sockaddr_in clientAddr, serverAddr;
	string task = "";
	
	//start up the server
	startUpServer(&listenfd, &serverAddr);
	
	while(true){
		clientLength = sizeof(clientAddr);
		connfd = Accept(listenfd, (SA *) &clientAddr, &clientLength);
	
		cout << "Here and listening" << endl;
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
			quit(connfd, listenfd);
		}
		Close(connfd);			 
	}
}

Semaphore& PtolemyServer::iterSem()
{
	static Semaphore sem_("iterate semaphore", 0);
	return sem_;
}


//Note that if you add this callback at the beginning of a task
//it is necessary to remove it at the end so it wont modify the static
//semephore in the future when it is needed again.
bool PtolemyServer::iter_callback(void *data)
{
	iterSem().up();
	return false;
}



string PtolemyServer::Iterate(vector<string> doOnce, int size1, vector<string> iterate, int size2, int numParams, string picPath, string picFormat)
{
	iterSem().down();
	
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

void PtolemyServer::processItrRequest(int sockfd)
{
	ssize_t n;   //return value of some network functions
	char line[MAXLINE];	  //lines we get sent
	char *retVal;	  //value we will return to client
	string rv = "none";	   //return value for a run
	
	string picPath, picFormat, temp;	//path were pictures will be saved
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
	/*
	string netName = "/scratch/test/test2.net";
	ASSERT(gui);
	string evalret = gui->eval("loadnet {" + netName + "}");
	cout << "eval " << evalret << endl;
	//cout << "Here and loaded" << endl;
	//gui->execute("loadnet {" + netName + "}");
	*/
	/*
	//iterSem().down();
	LoadNet *load = new LoadNet(gui);
	Thread *t = new Thread(load, "load a network", 0, Thread::NotActivated);
	t->setStackSize(1024*1024);
	t->activate(false);
	t->join();
	iterSem().down();
	*/
	rv = Iterate(input1, size1, input2, size2, numParams, picPath, picFormat);
	cout << "done iterating rv = " << rv << "??" << endl;
	
	retVal = ccast_unsafe(rv);
	Writen(sockfd, retVal, rv.size());
	
}

void PtolemyServer::quit(int sockfd,int listenfd)
{
	iterSem().down();
	
	char message[8] = "quiting";
	Writen(sockfd, message, 7);
	Close(sockfd);
	Close(listenfd);
	
	QuitSCIRun *quit = new QuitSCIRun(gui);

	Thread *t = new Thread(quit, "quit scirun", 0, Thread::NotActivated);
	t->setDaemon();
	t->setStackSize(1024*10);
	t->activate(false);
	t->detach();
	iterSem().down();
}
