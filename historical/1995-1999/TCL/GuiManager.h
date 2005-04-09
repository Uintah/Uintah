
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

#include <Classlib/Array1.h>
#include <TCL/Remote.h>
#include <Multitask/ITC.h>

class GuiManager {
	Mutex access;
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
 
#endif
