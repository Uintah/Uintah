/*
 *  GuiServer.h: Server running on Master SCIRun to service remote GUI
 *   requests.
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

#ifndef SCI_project_GuiServer_h
#define SCI_project_GuiServer_h 1

#include <Classlib/Array1.h>
#include <Multitask/Task.h>
#include <TCL/Remote.h>
#include <TCL/TCLTask.h>

class GuiServer : public Task {
    private:
	int gui_socket;
	Array1<int> clients;
    public:
	GuiServer ();
	~GuiServer();

    private:
	virtual int body (int);
	void getValue (char*, TCLMessage*);
};

#endif
