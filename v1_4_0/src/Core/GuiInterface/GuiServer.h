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

#include <Core/GuiInterface/Remote.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Thread/Runnable.h>
#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE GuiServer : public Runnable {
    private:
	int gui_socket;
        vector<int> clients;
    public:
	GuiServer ();
	~GuiServer();

    private:
	virtual void run();
	void getValue (char*, TCLMessage*);
};

} // End namespace SCIRun


#endif
