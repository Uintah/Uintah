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
 *  TCL.cc: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/Remote.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;
#include <tcl.h>
#include <tk.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>                             // defines read() and write()
#endif

extern "C" Tcl_Interp* the_interp;

namespace SCIRun {

struct TCLCommandData {
    TCL* object;
    void* userdata;
};

GuiManager *gm = NULL;

void set_guiManager (GuiManager *mgr)
{
    gm = mgr;
}
 
#if 0
static string prefix()
{
    static int haveit=0;
    static string pf;
    if(!haveit){
	char* p = getenv("PSE_WORK");
	if(p){
	    pf=string(p);
	} else {
		printf("Error: PSE_WORK variable not set!\n");
	    Task::exit_all(-1);
	}
    }
    return pf;
}

static string application()
{
    static int haveit=0;
    static string app;
    if(!haveit){
	char* a = getenv("APPLICATION");
	if(a){
	    app=string(a);
	} else {
		printf("Error; APPLICATION environment variable not set!\n");
	    Task::exit_all(-1);
	}
    }
    return app;
}

#endif
void TCL::execute(const string& str)
{
  GuiManager::execute(str);
}

int TCL::eval(const string& str, string& result)
{
  return GuiManager::eval(str, result);
}

void TCL::source_once(const string& filename)
{
  string result;
  if(!eval("source " + filename, result)) {
    char* msg = ccast_unsafe("Couldn't source file '" + filename + "'");
    Tcl_AddErrorInfo(the_interp,msg);
    Tk_BackgroundError(the_interp);
  }

#if 0
    int code;

    TCLTask::lock();

    string pse_filename(filename);

    char* fn = ccast_unsafe(pse_filename);
    code = Tcl_EvalFile(the_interp, fn);

    if(code != TCL_OK) {
      char msg[256];
      sprintf(msg,"Failed on loading %s err %d\n", fn, code);
      Tcl_AddErrorInfo(the_interp,msg);
      Tk_BackgroundError(the_interp);
    }

    TCLTask::unlock();
#endif
}

static int do_command(ClientData cd, Tcl_Interp*, int argc, char* argv[])
{
    TCLCommandData* td=(TCLCommandData*)cd;
    TCLArgs args(argc, argv);
    td->object->tcl_command(args, td->userdata);
    if(args.have_result_)
    {
      Tcl_SetResult(the_interp,
		    strdup(args.string_.c_str()),
		    (Tcl_FreeProc*)free);
    }
    return args.have_error_?TCL_ERROR:TCL_OK;
}

void TCL::add_command(const string& command, TCL* callback, void* userdata)
{
    TCLTask::lock();
    TCLCommandData* command_data=scinew TCLCommandData;
    command_data->object=callback;
    command_data->userdata=userdata;
    Tcl_CreateCommand(the_interp, ccast_unsafe(command),
		      do_command, command_data, 0);
    TCLTask::unlock();
}

void
TCL::delete_command( const string& command )
{
    TCLTask::lock();
    Tcl_DeleteCommand(the_interp, ccast_unsafe(command));
    TCLTask::unlock();
}

TCL::TCL()
{
}

TCL::~TCL()
{
}

void TCL::emit_vars(ostream& out, string& midx)
{
    for(int i=0;i<vars.size();i++)
      {
//	cerr << "emit: " << vars[i]->str() << endl;
        vars[i]->emit(out, midx);
      }
}


TCLArgs::TCLArgs(int argc, char* argv[])
: args_(argc)
{
    for(int i=0;i<argc;i++)
	args_[i] = string(argv[i]);
    have_error_ = false;
    have_result_ = false;
}

TCLArgs::~TCLArgs()
{
}

int TCLArgs::count()
{
    return args_.size();
}

string TCLArgs::operator[](int i)
{
    return args_[i];
}

void TCLArgs::error(const string& e)
{
    string_ = e;
    have_error_ = true;
    have_result_ = true;
}

void TCLArgs::result(const string& r)
{
    if(!have_error_){
	string_ = r;
	have_result_ = true;
    }
}

void TCLArgs::append_result(const string& r)
{
    if(!have_error_){
	string_ += r;
	have_result_ = true;
    }
}

void TCLArgs::append_element(const string& e)
{
    if(!have_error_){
	if(have_result_)
	    string_ += ' ';
	string_ += e;
	have_result_ = true;
    }
}

string TCLArgs::make_list(const string& item1, const string& item2)
{
    char* argv[2];
    argv[0]= ccast_unsafe(item1);
    argv[1]= ccast_unsafe(item2);
    char* ilist=Tcl_Merge(2, argv);
    string res(ilist);
    free(ilist);
    return res;
}

string TCLArgs::make_list(const string& item1, const string& item2,
			const string& item3)
{
    char* argv[3];
    argv[0]=ccast_unsafe(item1);
    argv[1]=ccast_unsafe(item2);
    argv[2]=ccast_unsafe(item3);
    char* ilist=Tcl_Merge(3, argv);
    string res(ilist);
    free(ilist);
    return res;
}

string TCLArgs::make_list(const Array1<string>& items)
{
    char** argv=scinew char*[items.size()];
    for(int i=0;i<items.size();i++)
    {
      argv[i]= ccast_unsafe(items[i]);
    }
    char* ilist=Tcl_Merge(items.size(), argv);
    string res(ilist);
    free(ilist);
    delete[] argv;
    return res;
}

void TCL::reset_vars()
{
    for(int i=0;i<vars.size();i++)
	vars[i]->reset();
}

void TCL::register_var(GuiVar* v)
{
    vars.add(v);
}

void TCL::unregister_var(GuiVar* v)
{
    for(int i=0;i<vars.size();i++){
	if(vars[i]==v){
	    vars.remove(i);
	    return;
	}
    }
}

int TCL::get_gui_stringvar(const string& base, const string& name,
			   string& value)
{
    string n(base + "-" + name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(n), TCL_GLOBAL_ONLY);
    if(!l){
	TCLTask::unlock();
	return 0;
    }
    value=l;
    TCLTask::unlock();
    return 1;
}

int TCL::get_gui_boolvar(const string& base, const string& name,
			 int& value)
{
    string n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(n), TCL_GLOBAL_ONLY);
    if(!l){
	TCLTask::unlock();
	return 0;
    }
    value=0;
    if(!strcmp(l, "yes")){
	value=1;
    } else if(!strcmp(l, "true")){
	value=1;
    } else if(!strcmp(l, "1")){
	value=1;
    }
    TCLTask::unlock();
    return 1;
}

int TCL::get_gui_doublevar(const string& base, const string& name,
			   double& value)
{
    string n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(n), TCL_GLOBAL_ONLY);
    if(!l){
	TCLTask::unlock();
	return 0;
    }
    char* end;
    value=strtod(l, &end);
    if(*end != 0){
	// Error reading the double....
	TCLTask::unlock();
	return 0;
    } else {
	TCLTask::unlock();
	return 1;
    }
}

int TCL::get_gui_intvar(const string& base, const string& name,
			 int& value)
{
    string n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, ccast_unsafe(n), TCL_GLOBAL_ONLY);
    if(!l){
	Tk_BackgroundError(the_interp);
	TCLTask::unlock();
	return 0;
    }
    char* end;
    value=(int)strtol(l, &end, 10);
    if(*end != 0){
	// Error reading the double....
	TCLTask::unlock();
	return 0;
    } else {
	TCLTask::unlock();
	return 1;
    }
}

void TCL::set_guivar(const string& base, const string& name,
		     const string& value)
{
    string n(base + "-" + name);
    TCLTask::lock();
    Tcl_SetVar(the_interp, ccast_unsafe(n), ccast_unsafe(value),
	       TCL_GLOBAL_ONLY);
    TCLTask::unlock();
}

} // End namespace SCIRun

