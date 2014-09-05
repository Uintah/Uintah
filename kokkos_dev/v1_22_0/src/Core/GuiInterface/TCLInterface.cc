/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


string TCLInterface::eval(const string& str)
{
  string result("");
  TCLTask::lock();
  int code = Tcl_Eval(the_interp, ccast_unsafe(str));
  if(code != TCL_OK){
    Tk_BackgroundError(the_interp);
  } else {
    result=string(the_interp->result);
  }
  TCLTask::unlock();
  return result;
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
  for( unsigned int cnt = 0; cnt < fixed_errmsg.size(); cnt++ )
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



bool
TCLInterface::complete_command(const string &command)
{
  const int len = command.length()+1;
  char *src = scinew char[len];
  strcpy (src, command.c_str());
  TCLTask::lock();
  Tcl_Parse parse;
  const int ret_val = Tcl_ParseCommand(0, src, len, 1, &parse);
  TCLTask::unlock();
  return (ret_val == TCL_OK);
}
