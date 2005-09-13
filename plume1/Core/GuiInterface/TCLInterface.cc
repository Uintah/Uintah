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
#include <Core/Util/Assert.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Thread.h>
#include <Core/Exceptions/GuiException.h>
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

#ifdef EXPERIMENTAL_TCL_THREAD
// in windows, we will use a different interface (since it hangs in many
// places the non-windows ones do not).  Instead of each thread potentially
// calling TCL code, and have the GUI lock, we will send the TCL commands
// to the tcl thread via a mailbox, and TCL will get the data via an event 
// callback
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Semaphore.h>
#include <string>
using std::string;

struct EventMessage {
  EventMessage(string command, Semaphore* sem) : command(command), sem(sem) {};
  string command;
  Semaphore* sem;
  int code;
  string result;
};

static Mailbox<EventMessage*> tclQueue("TCL command mailbox", 50);
static Tcl_Time tcl_time;

int eventCallback(Tcl_Event* event, int flags);


#endif

extern "C" Tcl_Interp* the_interp;

TCLInterface::TCLInterface()
{
  MemStats* memstats=scinew MemStats;
  memstats->init_tcl(this);
  
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");
  const char *itcl_dir = sci_getenv("SCIRUN_ITCL_WIDGETS");
  ASSERT(srcdir);
  ASSERT(itcl_dir);
  const std::string src(srcdir);
  const std::string itcl(itcl_dir);

  eval("set scirun2 0");
  eval("lappend auto_path "+src+"/Core/GUI "+src+"/Dataflow/GUI "+itcl);
}

TCLInterface::~TCLInterface()
{
}


void TCLInterface::execute(const string& str)
{
  string result("");
  eval(str, result);
}


string TCLInterface::eval(const string& str)
{
  string result("");
  eval(str, result);
  return result;
}


#ifndef EXPERIMENTAL_TCL_THREAD

int TCLInterface::eval(const string& str, string& result)
{
  TCLTask::lock();
  //cerr << " Evaling cmd: " << str << " from thread " << Thread::self()->getThreadName() << "\n";
  int code = Tcl_Eval(the_interp, ccast_unsafe(str));
  if(code != TCL_OK){
    Tk_BackgroundError(the_interp);
    result="";
  } else {
    result=string(the_interp->result);
  }
  //cerr << " Done evaling cmd: " << str << " from thread " << Thread::self()->getThreadName() << "\n";
  TCLTask::unlock();
  return code == TCL_OK;
}

#else
    
int TCLInterface::eval(const string& str, string& result)
{
  // if we are the TCL Thread, go ahead and execute the command, otherwise
  // add it to the queue so the tcl thread can get it later
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 ) {
    int code = Tcl_Eval(the_interp, ccast_unsafe(str));
    if(code != TCL_OK){
      Tk_BackgroundError(the_interp);
    } else {
      result = string(the_interp->result);
    }
    return code == TCL_OK;
  }
  else {
    Semaphore sem("wait for tcl", 0);
    EventMessage em(str, &sem);

    tclQueue.send(&em);

    sem.down();
    result = em.result;
    return em.code == TCL_OK;
  }
}


int eventCallback(Tcl_Event* event, int flags)
{
   EventMessage* em;
   while (tclQueue.tryReceive(em)) {
     int code = Tcl_Eval(the_interp, ccast_unsafe(em->command));
     if(code != TCL_OK){
       Tk_BackgroundError(the_interp);
     } else {
       em->result = string(the_interp->result);
     }
     em->code = code;
     em->sem->up();
   }
   return 1;
}

namespace SCIRun {
void eventSetup(ClientData cd, int flags)
{
  if (tclQueue.numItems() > 0) {
    tcl_time.sec = 0;
    tcl_time.usec = 0;
    Tcl_SetMaxBlockTime(&tcl_time);
  }
}
void eventCheck(ClientData cd, int flags)
{
  if (tclQueue.numItems() > 0) {
    Tcl_Event* event = scinew Tcl_Event;
    event->proc = eventCallback;
    Tcl_QueueEvent(event, TCL_QUEUE_TAIL);
  }
}
} // end namespace SCIRun
#endif


void TCLInterface::source_once(const string& filename)
{
  string result;
  if(!eval("source " + filename, result)) {
    char* msg = ccast_unsafe("Couldn't source file '" + filename + "'");
    Tcl_AddErrorInfo(the_interp,msg);
    Tk_BackgroundError(the_interp);
  }
}

static int do_command(ClientData cd, Tcl_Interp*, int argc, TCLCONST char* argv[])
{
  TCLCommandData* td=(TCLCommandData*)cd;
  GuiArgs args(argc, argv);
  try {
    td->object->tcl_command(args, td->userdata);
  } catch (const GuiException &exception) {
    args.string_ = exception.message();
    args.have_error_ = true;
    args.have_result_ = true;
  } catch (const string &message) {
    args.string_ = message;
    args.have_error_ = true;
    args.have_result_ = true;
  } catch (const char *message) {
    args.string_ = message;
    args.have_error_ = true;
    args.have_result_ = true;
  }

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
  TCLCONST char* l=Tcl_GetVar(the_interp, ccast_unsafe(name),
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
TCLInterface::get_map(const std::string& name, 
		      const std::string& key, 
		      std::string& value)
{
  TCLTask::lock();
  const char* l=Tcl_GetVar2(the_interp, 
			    ccast_unsafe(name),
			    ccast_unsafe(key),
			    TCL_GLOBAL_ONLY);
  value = l?l:"";
  TCLTask::unlock();
  return l;
}
		      
bool
TCLInterface::set_map(const std::string& name, 
		      const std::string& key, 
		      const std::string& value)
{
  TCLTask::lock();
  const char *l = Tcl_SetVar2(the_interp, 
			      ccast_unsafe(name),
			      ccast_unsafe(key),
			      ccast_unsafe(value), 
			      TCL_GLOBAL_ONLY);
  TCLTask::unlock();
  return l;
}


bool
TCLInterface::extract_element_from_list(const std::string& contents, 
					const vector <int>& indexes,
					std::string& value)
{
  string command = "lsubindex {"+contents+"}";
  const unsigned int last = indexes.size()-1;
  for (unsigned int i = 0; i <= last; ++i)
    command += " "+to_string(indexes[last-i]);
  return eval(command, value);
}


bool
TCLInterface::set_element_in_list(std::string& contents, 
				  const vector <int>& indexes,
				  const std::string& value)
{
  string command = "lreplacesubindex {"+contents+"} "+value;
  const unsigned int last = indexes.size()-1;
  for (unsigned int i = 0; i <= last; ++i)
    command += " "+to_string(indexes[last - i]);
  return eval(command, contents);
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
  delete[] src;
  return (ret_val == TCL_OK);
}
