
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

#include <SCICore/share/share.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/Multitask/Task.h>
#include <SCICore/TclInterface/Remote.h>
#include <SCICore/TclInterface/TCLTask.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Multitask::Task;
using SCICore::Containers::Array1;

class SCICORESHARE GuiServer : public Task {
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

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:14  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:23  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:32  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
