//static char *id="@(#) $Id$";

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

#ifndef _WIN32
#include <SCICore/TclInterface/GuiServer.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <tcl.h>
#include <tk.h>

//#define DEBUG 1
extern "C" Tcl_Interp* the_interp;

namespace SCICore {
namespace TclInterface {

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
	    	        return;
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

} // End namespace TclInterface
} // End namespace SCICore
#endif // win32

//
// $Log$
// Revision 1.6  1999/09/08 02:26:55  sparker
// Various #include cleanups
//
// Revision 1.5  1999/08/28 17:54:51  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/23 06:30:39  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.3  1999/08/18 20:20:21  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:39:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:14  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
