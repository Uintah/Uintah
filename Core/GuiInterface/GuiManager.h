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

#include <Core/Containers/Array1.h>
#include <Core/GuiInterface/Remote.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {


class SCICORESHARE GuiManager {
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

} // End namespace SCIRun

 
#endif
