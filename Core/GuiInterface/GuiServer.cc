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
 *  GuiServer.cc: Server running on Master Dataflow to service remote GUI
 *   requests
 *
 *  Sets up a listening socket for incoming client requests.
 *  Takes in a request to get a TCL value or execute a TCL string.  Calls
 *  the correct TCL function directly.  Design choice: invoke the skeleton
 *  method.
 *
 *  Written by:
 *   Michelle Miller
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef _WIN32
#include <Core/Containers/Array1.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiServer.h>
#include <unistd.h>
#include <sys/types.h>
#include <tcl.h>
#include <tk.h>

#include <iostream>
#include <string>

using namespace std;

#define DEBUG 0
extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

GuiServer::GuiServer()
    : gui_socket(0)
{
}

GuiServer::~GuiServer()
{
}

void
GuiServer::run()
{
    char buf[BUFSIZE];
    fd_set readfds;
    TCLMessage msg;
    
    // open up a listening socket and keep it open to allow dynamic connections
    int listen_socket = setupConnect (BASE_PORT - 1);
    FD_ZERO (&readfds);

    // add listen socket to set of reading filedes for select
    FD_SET (listen_socket, &readfds);
    int maxfd = listen_socket;

#if DEBUG
cerr <<"GuiServer::getClients() opened listen_socket = "<< listen_socket<<endl;
#endif

    for (;;) {

#if DEBUG
cerr << "GuiServer::getClients() before execute of select(), readfds = "
     << readfds << endl;
#endif

	// s has number of fds ready to read, readfds contains only ready fds
#if DEBUG
int s = 
#endif
    select (maxfd + 1, &readfds, 0, 0, 0);

#if DEBUG
cerr << "GuiServer::getClients() select() returned" << s << "sockets ready, ";
cerr << "readfds = " << readfds << endl;
cerr << "GuiServer::getClients() clients.size() = " << clients.size() << endl;
#endif

	// read from all available clients 
      	for (unsigned int i = 0; i < clients.size(); i++)  {
	    if (FD_ISSET (clients[i], &readfds)) {

#if DEBUG
cerr << "GuiServer::getClients() read from clients["<<i<<"], socket = " <<
         clients[i]<< endl;
#endif

		// assume one read will get entire message
		bzero (buf, sizeof (buf));
		if (read (clients[i], buf, sizeof(buf)) < 0) {
		    perror ("reading client socket message");
		}
		getValue (buf, &msg);

		// execute should NOT reply!  This is the bug!
                if (msg.f != exec)  {
		    // write the message back with the newly acquired value
        	    bzero (buf, sizeof (buf));
        	    bcopy ((char *) &msg, buf, sizeof (msg));
        	    if (write (clients[i], buf, sizeof(buf)) < 0) {
	    	        perror ("writing to client socket");
	    	        return;
		    }
 		}

	    }
	    FD_SET (clients[i], &readfds);
	}

	// check for a pending connection request
	if (FD_ISSET (listen_socket, &readfds)) {
		int new_fd = acceptConnect (listen_socket);
		clients.push_back(new_fd);
#if DEBUG
cerr << "GuiServer::getClients() add new client connection " << new_fd << endl;
#endif
	  	FD_SET (new_fd, &readfds);
	    	if (new_fd > maxfd)
		    maxfd = new_fd;
       	}

	// reset listen_socket fd to be checked by select() in next pass
    	FD_SET (listen_socket, &readfds);
    }
}

void
GuiServer::getValue (char* buffer, TCLMessage* msg)
{
    bcopy (buffer, (char *) msg, sizeof (*msg));

#if DEBUG
cerr << "GuiServer::getValue(): called by variable = " << msg->tclName << endl;
#endif

    switch (msg->f) {
       	case getDouble: {
            TCLTask::lock();
            char* l=Tcl_GetVar(the_interp, msg->tclName,TCL_GLOBAL_ONLY);
            if(l){
             	Tcl_GetDouble(the_interp, l, &(msg->un.tdouble));
            }
            TCLTask::unlock();

#if DEBUG
cerr <<"GuiServer::getValue(): double = " << msg->un.tdouble << endl; 
#endif
	    break;
 	}
	case getInt: {
            TCLTask::lock();
            char* l=Tcl_GetVar(the_interp, msg->tclName,TCL_GLOBAL_ONLY);
            if(l){
             	Tcl_GetInt(the_interp, l, &(msg->un.tint));
            }
            TCLTask::unlock();

#if DEBUG
cerr << "GuiServer::getValue(): int = " << msg->un.tint << endl; 
#endif
	    break;
	}
	case getString: {
            TCLTask::lock();
            char* l=Tcl_GetVar(the_interp, msg->tclName,TCL_GLOBAL_ONLY);
            if(!l){
              	l="";
            }
            strcpy (msg->un.tstring, l);

#if DEBUG
cerr << "GuiServer::getValue(): string = " << msg->un.tstring << endl; 
#endif
            TCLTask::unlock();
	    break;
	}
	// does not update state in skeleton
	case exec: {
	    TCLTask::lock();
            int code = Tcl_Eval(the_interp, msg->un.tstring);
            if(code != TCL_OK)
           	Tk_BackgroundError(the_interp);
       	    TCLTask::unlock();
	    break;
	}
	default:
	    break;
    } 
}

} // End namespace SCIRun
#endif // win32
