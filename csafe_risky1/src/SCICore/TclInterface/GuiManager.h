
/*
 *  GuiManager.h: Client side (slave) manager of a pool of remote GUI
 *   connections
 *
 *  This class keeps a dynamic array of connections for use by TCL variables
 *  needing to get their values from the Master.  These are kept in a pool.
 *
 *  Written by:
 *   Michelle Miller
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_project_GuiManager_h
#define SCI_project_GuiManager_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/TclInterface/Remote.h>
#include <SCICore/Thread/Mutex.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::Array1;

class SCICORESHARE GuiManager {
    SCICore::Thread::Mutex access;
    private:
	Array1<int> connect_pool;	// available sockets
	char host[HOSTNAME];
	int base_port;
    public:
	GuiManager (char* host, char* portname);
	~GuiManager();
	int addConnection();
	int getConnection();
	void putConnection (int sock);
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/28 17:54:51  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:42  sparker
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
