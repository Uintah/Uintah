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

#ifndef GuiManager_h
#define GuiManager_h 

#include <string>
#include <map>
#include <pair.h>


namespace SCIRun {

using std::string;
using std::map;
using namespace std;

class Part;
class TCL;
class TCLArgs;
typedef void *ClientData;

class GuiManager {
public:
  GuiManager () {}
  virtual ~GuiManager() {}

  virtual void add_text( const string &) = 0;
  virtual void post_msg( const string &, bool err=true) = 0;
  virtual string get(string& value, string varname, int& is_reset) = 0;
  virtual void set(string& value, string varname, int& is_reset) = 0;
  
  virtual double get(double& value, string varname,int& is_reset) = 0;
  virtual void set(double& value, string varname,int& is_reset) = 0;
  
  virtual int get(int& value, string varname, int& is_reset) = 0;
  virtual void set(int& value, string varname, int& is_reset) = 0;
  
  virtual void execute(const string& str) = 0;
  virtual void eval(const string& str, string& result) = 0;
  
  virtual void add_command(const string& command, Part* callback, void*) = 0;
  virtual void add_command(const string& command, TCL* callback, void*) = 0;
  virtual void delete_command( const string &command ) = 0;


  // To get at tcl variables
  virtual int get_gui_stringvar(const string &, const string &, string & ) = 0;
  virtual int get_gui_boolvar(const string &, const string &, int & ) = 0;
  virtual int get_gui_doublevar(const string &, const string &, double & ) = 0;
  virtual int get_gui_intvar(const string &, const string &, int & ) = 0;
  virtual void set_guivar(const string &, const string &, const string& ) = 0;

  virtual char *merge( int, char **) = 0;

  virtual void lock() = 0;
  virtual void unlock() = 0;
  
  // Stream
  virtual void create_var( const string & ) = 0;
  virtual void remove_var( const string & ) = 0;
  virtual void set_var( const string &, const string &str ) = 0;

  virtual void *name_to_window( const string &name ) = 0;
  virtual void *get_glx( const string &name ) = 0;
  virtual int  query_OpenGL() = 0;
};

extern GuiManager *gm;
} // End namespace SCIRun

 
#endif

