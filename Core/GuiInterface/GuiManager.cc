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
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <string.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdlib.h>	// needed for atoi
#include <iostream>

#include <tcl.h>

using namespace std;

extern "C" Tcl_Interp* the_interp;
namespace SCIRun {

GuiManager* GuiManager::gm_ = 0;
Mutex GuiManager::gm_lock_("GuiManager: static instance");


GuiManager::GuiManager()
    : access ("GUI manager access lock")
{
  
}

GuiManager::~GuiManager()
{
}

GuiManager& GuiManager::getGuiManager() {
  if(gm_ == 0) {
    gm_lock_.lock();
    if(gm_ == 0) {
      gm_ = new GuiManager;
    }
    gm_lock_.unlock();
  }
  return *gm_;
  
}

string GuiManager::get(string& value, string varname, int& is_reset) {
  if(is_reset) {
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(varname),
		       TCL_GLOBAL_ONLY);
    if(!l){
      l="";
    }
    value=string(l);
    is_reset=0;
    TCLTask::unlock();
  }
    //cerr << "GuiString get: " << varname << " to " << value << endl;
  return value;

}
void GuiManager::set(string& value, string varname, int &is_reset) {
  is_reset=0;
  TCLTask::lock();
  Tcl_SetVar(the_interp, ccast_unsafe(varname),
	     ccast_unsafe(value), TCL_GLOBAL_ONLY);
  TCLTask::unlock();
  //cerr << "GuiString set: " << varname << " to " << value << endl;
}

double GuiManager::get(double& value, string varname, int& is_reset) {
  if(is_reset) {
    TCLTask::lock(); 
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(varname),
		       TCL_GLOBAL_ONLY); 
    if(l){ 
      Tcl_GetDouble(the_interp, l, &value);
      is_reset=0; 
    } 
    TCLTask::unlock();
  }
  //cerr << "GuiDouble get: " << varname << " to " << value << endl;
  return value; 
}
void GuiManager::set(double& value, string varname, int& is_reset) {
  is_reset = 0;
  TCLTask::lock(); 
  char buf[50]; 
  sprintf(buf, "%g", value); 
  
  Tcl_SetVar(the_interp, ccast_unsafe(varname), 
	     buf, TCL_GLOBAL_ONLY); 
  TCLTask::unlock(); 
  //cerr << "GuiDouble set: " << varname << " to " << value << endl;
}

int GuiManager::get(int& value, string varname, int &is_reset) {
  if(is_reset) {
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(varname),
		       TCL_GLOBAL_ONLY);
    if(l){
      Tcl_GetInt(the_interp, l, &value);
      is_reset=0;
    }
    TCLTask::unlock();
  }
  //cerr << "GuiInt get: " << varname << " to " << value << endl;
  return value;
}

void GuiManager::set(int& value, string varname, int &is_reset) {
  is_reset=0;
  TCLTask::lock();
  char buf[20];
  sprintf(buf, "%d", value);
  Tcl_SetVar(the_interp, ccast_unsafe(varname), buf, TCL_GLOBAL_ONLY);
  TCLTask::unlock();
  //cerr << "GuiInt set: " << varname << " to " << value << endl;
}


} // End namespace SCIRun

#endif
