/*
 *  GuiServer.cc: Server running on Master SCIRun to service remote GUI
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

#include <TCL/GuiServer.h>

#include <Classlib/Array1.h>
#include <Multitask/Task.h>
#include <TCL/TCLTask.h>

#include <sys/types.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>

#include <strings.h>

//#define DEBUG 1
extern Tcl_Interp* the_interp;
extern int acceptConnect (int socket);
extern int setupConnect (int port);

GuiServer::GuiServer() : Task("GuiServer"), gui_socket(0)
{}

GuiServer::~GuiServer()
{}

int
GuiServer::body(int)
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

#ifdef DEBUG
cerr <<"GuiServer::getClients() opened listen_socket = "<< listen_socket<<endl;
#endif

    for (;;) {

#ifdef DEBUG
cerr << "GuiServer::getClients() before execute of select(), readfds = "
     << readfds << endl;
#endif

	// s has number of fds ready to read, readfds contains only ready fds
#ifdef DEBUG
	int s = 
#endif
	   select (maxfd + 1, &readfds, 0, 0, 0);

#ifdef DEBUG
cerr << "GuiServer::getClients() select() returned" << s << "sockets ready, ";
cerr << "readfds = " << readfds << endl;
cerr << "GuiServer::getClients() clients.size() = " << clients.size() << endl;
#endif

	// read from all available clients 
      	for (int i = 0; i < clients.size(); i++)  {
	    if (FD_ISSET (clients[i], &readfds)) {

#ifdef DEBUG
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
	    	        return -1;
		    }
 		}

	    }
	    FD_SET (clients[i], &readfds);
	}

	// check for a pending connection request
	if (FD_ISSET (listen_socket, &readfds)) {
		int new_fd = acceptConnect (listen_socket);
		clients.add(new_fd);
#ifdef DEBUG
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

#ifdef DEBUG
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

#ifdef DEBUG
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

#ifdef DEBUG
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

#ifdef DEBUG
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
