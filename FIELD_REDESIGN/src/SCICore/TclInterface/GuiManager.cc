//static char *id="@(#) $Id$";

/*
 *  GuiManager.cc: Client side (slave) manager of a pool of remote GUI
 *   connections.
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

#ifndef _WIN32

#include <SCICore/TclInterface/GuiManager.h>
#include <SCICore/Containers/Array1.h>

#include <string.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>	// needed for atoi
#include <iostream>
using namespace std;
namespace SCICore {
namespace TclInterface {

GuiManager::GuiManager (char* hostname, char* portname) 
    : access("GUI manager access lock")
{
    base_port = atoi (portname);
    strcpy (host, hostname);
    strcat (host,".cs.utah.edu");
}

GuiManager::~GuiManager()
{ }

int
GuiManager::addConnection()
{
    // open one socket 
    int conn = requestConnect (base_port-1, host);
    if (conn == -1) {
            perror ("outgoing connection setup failed");
            return -1;
    }
    connect_pool.add(conn);
    return conn;
}

/*   Traverse array of sockets looking for an unused socket.  If no unused
 *   sockets, add another socket, lock it, and return to caller.
 */
int
GuiManager::getConnection()
{
    int sock;

#ifdef DEBUG
cerr << "attempting to lock, lock = " << &access << " pid " << getpid() << endl;
#endif

    access.lock();

#ifdef DEBUG
cerr << "GuiManager::getConnection() pid " << getpid() << " is locking"
     << endl;
#endif

    if (connect_pool.size() == 0) {
	sock = addConnection();
    } else {
	sock = connect_pool[connect_pool.size()-1];	// get last elem
    }

    connect_pool.remove(connect_pool.size()-1);	// take out of pool

#ifdef DEBUG
cerr << "GuiManager::getConnection() pid " << getpid() << " is unlocking"
     << endl;
#endif

    access.unlock();
    return sock;
}

/*  Return a connection to the pool.  */
void 
GuiManager::putConnection (int sock)
{
    access.lock();
    connect_pool.add(sock);
    access.unlock();
}

} // End namespace TclInterface
} // End namespace SCICore

#endif


//
// $Log$
// Revision 1.3.2.3  2000/10/26 17:42:53  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.4  2000/08/02 21:56:02  jas
// Added missing iostream and changed some debugging so that fd_set would
// not be printed out.
//
// Revision 1.3  1999/08/28 17:54:51  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:42  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:13  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
