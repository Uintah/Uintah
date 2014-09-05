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
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Time.h>
#include <Core/Exceptions/GuiException.h>
#include <tcl.h>
#include <tk.h>
#include <iostream>
#include <string>

// find a more 'consolidated' place to put this...
#if (TCL_MINOR_VERSION >= 4)
#define TCLCONST const
#else
#define TCLCONST
#endif

//#  define EXPERIMENTAL_TCL_THREAD

#ifdef _WIN32
#  include <windows.h>
#  undef SCISHARE
#  define SCISHARE __declspec(dllexport)
#  define EXPERIMENTAL_TCL_THREAD
#else
#  define SCISHARE
#endif

using namespace SCIRun;
using std::string;


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
static Mailbox<EventMessage*> tclQueue("TCL command mailbox", 50);
static Tcl_Time tcl_time = {0,0};
static Tcl_ThreadId tclThreadId = 0;
int eventCallback(Tcl_Event* event, int flags);
SCISHARE void eventSetup(ClientData cd, int flags);
SCISHARE void eventCheck(ClientData cd, int flags);

#endif

extern "C" {
  SCISHARE Tcl_Interp* the_interp;
}

TCLInterface::TCLInterface() :
  pause_semaphore_(new Semaphore("TCL Task Pause Semaphore",0)),
  paused_(0)
                   
{
  MemStats* memstats=scinew MemStats;
  memstats->init_tcl(this);
  
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");
  const char *itcl_dir = sci_getenv("SCIRUN_ITCL_WIDGETS");
  ASSERT(srcdir);
  ASSERT(itcl_dir);
  const std::string src(srcdir);
  const std::string itcl(itcl_dir);

  eval("set scijump 0");
  eval("lappend auto_path {" + src + "/Core/GUI} {" + src + "/Dataflow/GUI} {" + itcl + "}");
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

void
TCLInterface::unpause()
{
  if (paused_)
    pause_semaphore_->up();
  paused_ = false;
}

void
TCLInterface::real_pause()
{
  paused_ = true;
  pause_semaphore_->down();
}


#ifndef EXPERIMENTAL_TCL_THREAD

void
TCLInterface::pause()
{
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
    CommandEventMessage em(str);
    tclQueue.send(&em);
    Tcl_ThreadAlert(tclThreadId);
    em.wait_for_message_delivery();
    result = em.result();
    return em.code() == TCL_OK;
  }
}


void
TCLInterface::pause()
{
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 )
    return;
    
  PauseEventMessage em(this);
  tclQueue.send(&em);
  Tcl_ThreadAlert(tclThreadId);
  em.wait_for_message_delivery();
}


int eventCallback(Tcl_Event* event, int flags)
{
   EventMessage* em;
   while (tclQueue.tryReceive(em)) {

     PauseEventMessage *pause = dynamic_cast<PauseEventMessage *>(em);
     CommandEventMessage *command = dynamic_cast<CommandEventMessage *>(em);
     GetEventMessage *gm = dynamic_cast<GetEventMessage *>(em);
     SetEventMessage *sm = dynamic_cast<SetEventMessage *>(em);
     if (pause) {
       TCLInterface *tcl_interface = pause->tcl_interface_;
       pause->mark_message_delivered();
       tcl_interface->real_pause();
     }

     else if (command) {
       int code = Tcl_Eval(the_interp, ccast_unsafe(command->command()));
       if(code != TCL_OK){
         Tk_BackgroundError(the_interp);
       } else {
         command->result() = the_interp->result;
       }
       command->code() = code;
       command->mark_message_delivered();
     }
     else if (gm) {
       gm->execute();
       gm->mark_message_delivered();
     }
     else if (sm) {
       sm->execute();
       sm->mark_message_delivered();
     }
   }
   return 1;
}



void eventSetup(ClientData cd, int flags)
{
  // set thread id if not set
  tclThreadId = Tcl_GetCurrentThread();
  if (tclQueue.numItems() > 0) {
    Tcl_SetMaxBlockTime(&tcl_time);
  }
}

void eventCheck(ClientData cd, int flags)
{
  if (tclQueue.numItems() > 0) {
    Tcl_Event* event = scinew Tcl_Event;
    event->proc = eventCallback;
    Tcl_QueueEvent(event, TCL_QUEUE_HEAD);
  }
}

#endif

namespace SCIRun {

void TCLInterface::source_once(const string& filename)
{
  string result;
  if(!eval("source {" + filename + "}", result)) {
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

void TCLInterface::post_message(const string& errmsg, bool err)
{
  // "Status" could also be "warning", but this function only takes a
  // bool.  In the future, perhas we should update the functions that
  // call post_message to be more expressive.
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
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
  TCLCONST char* l=Tcl_GetVar(the_interp, ccast_unsafe(name),
		     TCL_GLOBAL_ONLY);
  value = l?l:"";
  TCLTask::unlock();
  return l;
#else
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 ) {
    TCLCONST char* l=Tcl_GetVar(the_interp, ccast_unsafe(name),
		       TCL_GLOBAL_ONLY);
    value = l?l:"";
    return l;
  }
  else {
    return get_map(name, "", value); // this version will parse out the key
  }
#endif
}

void
TCLInterface::set(const std::string& name, const std::string& value)
{
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
  Tcl_SetVar(the_interp, ccast_unsafe(name),
	     ccast_unsafe(value), TCL_GLOBAL_ONLY);
  TCLTask::unlock();
#else
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 ) {
    Tcl_SetVar(the_interp, ccast_unsafe(name),
	     ccast_unsafe(value), TCL_GLOBAL_ONLY);
  }
  else {
    set_map(name, "", value);
  }
#endif
}



bool
TCLInterface::get_map(const std::string& name, 
		      const std::string& key, 
		      std::string& value)
{
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
  const char* l=Tcl_GetVar2(the_interp, 
			    ccast_unsafe(name),
			    ccast_unsafe(key),
			    TCL_GLOBAL_ONLY);
  value = l?l:"";
  TCLTask::unlock();
  return l;
#else
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 ) {
    const char* l=Tcl_GetVar2(the_interp, 
			      ccast_unsafe(name),
			      ccast_unsafe(key),
			      TCL_GLOBAL_ONLY);
    value = l?l:"";
  }
  else {
    GetEventMessage em(name, key);
    tclQueue.send(&em);
    Tcl_ThreadAlert(tclThreadId);
    em.wait_for_message_delivery();
    value = em.result();
  }
  return value != "";
#endif
}
		      
bool
TCLInterface::set_map(const std::string& name, 
		      const std::string& key, 
		      const std::string& value)
{
#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
  const char *l = Tcl_SetVar2(the_interp, 
			      ccast_unsafe(name),
			      ccast_unsafe(key),
			      ccast_unsafe(value), 
			      TCL_GLOBAL_ONLY);
  TCLTask::unlock();
  return l;
#else
  if (strcmp(Thread::self()->getThreadName(), "TCL main event loop") == 0 ) {
    const char *l = Tcl_SetVar2(the_interp, 
			        ccast_unsafe(name),
			        ccast_unsafe(key),
			        ccast_unsafe(value), 
			        TCL_GLOBAL_ONLY);
    return l;
  }
  else {
    SetEventMessage em(name, value, key);
    tclQueue.send(&em);
    Tcl_ThreadAlert(tclThreadId);
    em.wait_for_message_delivery();
    return em.code() == TCL_OK;
  }
#endif
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


EventMessage::EventMessage() :
  result_(),
  return_code_(0),
  delivery_semaphore_(new Semaphore("TCL EventMessage Delivery",0))
{}

EventMessage::~EventMessage()
{
  if (delivery_semaphore_) delete delivery_semaphore_;
}

void
EventMessage::wait_for_message_delivery()
{
  delivery_semaphore_->down();
}

void
EventMessage::mark_message_delivered()
{
  delivery_semaphore_->up();
}

void GetEventMessage::execute()
{
  const char* l;
  if (key_ == "") {
    l = Tcl_GetVar(the_interp, ccast_unsafe(var_), TCL_GLOBAL_ONLY);
  }
  else {
    l = Tcl_GetVar2(the_interp, ccast_unsafe(var_), ccast_unsafe(key_),
			              TCL_GLOBAL_ONLY);
  }
  result_ = l?l:"";
  this->return_code_ = TCL_OK;
}

void SetEventMessage::execute()
{
  if (key_ == "") {
    Tcl_SetVar(the_interp, ccast_unsafe(var_), ccast_unsafe(val_), TCL_GLOBAL_ONLY);
    return_code_ = TCL_OK;
  }
  else {
    if (Tcl_SetVar2(the_interp, ccast_unsafe(var_), ccast_unsafe(key_),
			              ccast_unsafe(val_), TCL_GLOBAL_ONLY))
      return_code_ = TCL_OK;
    else
      return_code_ = !TCL_OK;
  }

}
}
