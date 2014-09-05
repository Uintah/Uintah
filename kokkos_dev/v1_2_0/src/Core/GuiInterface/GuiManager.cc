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

#include <Core/GuiInterface/GuiManager.h>
#include <Core/Containers/Array1.h>

#include <string.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>	// needed for atoi
#include <iostream>
using namespace std;
namespace SCIRun {

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

} // End namespace SCIRun

#endif
