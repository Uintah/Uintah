
/*
 *  GuiServer.h: Server running on Master Dataflow to service remote GUI
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

#include <Core/share/share.h>

#include <Core/Containers/Array1.h>
#include <Core/GuiInterface/Remote.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Thread/Runnable.h>

namespace SCIRun {


class SCICORESHARE GuiServer : public Runnable {
    private:
	int gui_socket;
	Array1<int> clients;
    public:
	GuiServer ();
	~GuiServer();

    private:
	virtual void run();
	void getValue (char*, TCLMessage*);
};

} // End namespace SCIRun


#endif
