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
 *  TCLInterface.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/GuiInterface/TCLInterface.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/MemStats.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <tcl.h>
#include <tk.h>
#include <iostream>

using namespace SCIRun;

namespace SCIRun {
  struct TCLCommandData {
    GuiCallback* object;
    void* userdata;
  };
}

extern "C" Tcl_Interp* the_interp;

TCLInterface::TCLInterface()
{
  MemStats* memstats=scinew MemStats;
  memstats->init_tcl(this);
}

TCLInterface::~TCLInterface()
{
}

void TCLInterface::execute(const string& str)
{
  TCLTask::lock();
  int code = Tcl_Eval(the_interp, ccast_unsafe(str));
  if(code != TCL_OK)
    Tk_BackgroundError(the_interp);
  TCLTask::unlock();
}

int TCLInterface::eval(const string& str, string& result)
{
  TCLTask::lock();
  int code = Tcl_Eval(the_interp, ccast_unsafe(str));
  if(code != TCL_OK){
    Tk_BackgroundError(the_interp);
    result="";
  } else {
    result=string(the_interp->result);
  }
  TCLTask::unlock();
  return code == TCL_OK;
}

void TCLInterface::source_once(const string& filename)
{
  string result;
  if(!eval("source " + filename, result)) {
    char* msg = ccast_unsafe("Couldn't source file '" + filename + "'");
    Tcl_AddErrorInfo(the_interp,msg);
    Tk_BackgroundError(the_interp);
  }
}

static int do_command(ClientData cd, Tcl_Interp*, int argc, char* argv[])
{
  TCLCommandData* td=(TCLCommandData*)cd;
  GuiArgs args(argc, argv);
  td->object->tcl_command(args, td->userdata);
  if(args.have_result_) {
    Tcl_SetResult(the_interp,
		  strdup(args.string_.c_str()),
		  (Tcl_FreeProc*)free);
  }
  return args.have_error_?TCL_ERROR:TCL_OK;
}

void TCLInterface::add_command(const string&command , GuiCallback* callback,
			       void* userdata)
{
  TCLTask::lock();
  TCLCommandData* command_data=scinew TCLCommandData;
  command_data->object=callback;
  command_data->userdata=userdata;
  Tcl_CreateCommand(the_interp, ccast_unsafe(command),
		    &do_command, command_data, 0);
  TCLTask::unlock();
}

void TCLInterface::delete_command( const string& command )
{
  TCLTask::lock();
  Tcl_DeleteCommand(the_interp, ccast_unsafe(command));
  TCLTask::unlock();
}

void TCLInterface::lock()
{
  TCLTask::lock();
}

void TCLInterface::unlock()
{
  TCLTask::unlock();
}

GuiContext* TCLInterface::createContext(const string& name)
{
  return new GuiContext(this, name);
}

void TCLInterface::postMessage(const string& errmsg, bool err)
{
  // "Status" could also be "warning", but this function only takes a
  // bool.  In the future, perhas we should update the functions that
  // call postMessage to be more expressive.
  // displayErrorWarningOrInfo() is a function in NetworkEditor.tcl.

  string status = "info";
  if( err ) status = "error";

  // Replace any double quotes (") with single quote (') as they break the
  // tcl parser.
  string fixed_errmsg = errmsg;
  for( int cnt = 0; cnt < fixed_errmsg.size(); cnt++ )
    {
      char ch = errmsg[cnt];
      if( ch == '"' )
	ch = '\'';
      fixed_errmsg[cnt] = ch;
    }
  string command = "displayErrorWarningOrInfo \"" + fixed_errmsg + "\" " + status;
  execute( command );
}

bool
TCLInterface::get(const std::string& name, std::string& value)
{
  TCLTask::lock();
  char* l=Tcl_GetVar(the_interp, ccast_unsafe(name),
		     TCL_GLOBAL_ONLY);
  if(!l){
    value="";
    TCLTask::unlock();
    return false;
  }
  value=l;
  TCLTask::unlock();
  return true;
}

void
TCLInterface::set(const std::string& name, const std::string& value)
{
  TCLTask::lock();
  Tcl_SetVar(the_interp, ccast_unsafe(name),
	     ccast_unsafe(value), TCL_GLOBAL_ONLY);
  TCLTask::unlock();
}
