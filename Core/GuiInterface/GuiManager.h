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

#include <Core/Thread/Mutex.h>
#include <string>

using std::string;

namespace SCIRun {


class SCICORESHARE GuiManager {
    Mutex access;
    private:
        static GuiManager *gm_;
        static Mutex       gm_lock_;
    public:
        GuiManager ();
	~GuiManager();
        static string get(string& value, string varname, int& is_reset);
        static void set(string& value, string varname, int& is_reset);

        static double get(double& value, string varname,int& is_reset);
        static void set(double& value, string varname,int& is_reset);

        static int get(int& value, string varname, int& is_reset);
        static void set(int& value, string varname, int& is_reset);

	static GuiManager& getGuiManager();
  
};

} // End namespace SCIRun

 
#endif
