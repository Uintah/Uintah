
/*
 *  TCL.h: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Malloc/Allocator.h>
#include <Multitask/Task.h>
#include <TCL/TCL.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>
#include <stdlib.h>
#include <string.h>

extern Tcl_Interp* the_interp;

struct TCLCommandData {
    TCL* object;
    void* userdata;
};

static clString prefix()
{
    static int haveit=0;
    static clString pf;
    if(!haveit){
	char* p;
	if(p=getenv("SCI_WORK")){
	    pf=clString(p);
	} else {
	    cerr << "Error: SCI_WORK variable not set!\n";
	    Task::exit_all(-1);
	}
    }
    return pf;
}

void TCL::execute(const clString& string)
{
    TCLTask::lock();
    int code = Tcl_Eval(the_interp, string());
    if(code != TCL_OK)
	Tk_BackgroundError(the_interp);
    TCLTask::unlock();
}

void TCL::execute(char* string)
{
    TCLTask::lock();
    int code = Tcl_Eval(the_interp, string);
    if(code != TCL_OK)
	Tk_BackgroundError(the_interp);
    TCLTask::unlock();
}

void TCL::source_once(const clString& filename)
{
    TCLTask::lock();
    clString complete_filename(prefix()+"/"+filename);
    int code = Tcl_EvalFile(the_interp, complete_filename());
    if(code != TCL_OK)
	Tk_BackgroundError(the_interp);
    TCLTask::unlock();
}

static int do_command(ClientData cd, Tcl_Interp*, int argc, char* argv[])
{
    TCLCommandData* td=(TCLCommandData*)cd;
    TCLArgs args(argc, argv);
    td->object->tcl_command(args, td->userdata);
    if(args.have_result)
	Tcl_SetResult(the_interp, strdup(args.string()), (Tcl_FreeProc*)free);
    return args.have_error?TCL_ERROR:TCL_OK;
}

void TCL::add_command(const clString& command, TCL* callback, void* userdata)
{
    TCLTask::lock();
    TCLCommandData* command_data=scinew TCLCommandData;
    command_data->object=callback;
    command_data->userdata=userdata;
    Tcl_CreateCommand(the_interp, command(), do_command, command_data, 0);
    TCLTask::unlock();
}

TCL::TCL()
{
}

TCL::~TCL()
{
}


TCLArgs::TCLArgs(int argc, char* argv[])
: args(argc)
{
    for(int i=0;i<argc;i++)
	args[i]=clString(argv[i]);
    have_error=0;
    have_result=0;
}

TCLArgs::~TCLArgs()
{
}

int TCLArgs::count()
{
    return args.size();
}

clString TCLArgs::operator[](int i)
{
    return args[i];
}

void TCLArgs::error(const clString& e)
{
    string=e;
    have_error=1;
    have_result=1;
}

void TCLArgs::result(const clString& r)
{
    if(!have_error){
	string=r;
	have_result=1;
    }
}

void TCLArgs::append_result(const clString& r)
{
    if(!have_error){
	string+=r;
	have_result=1;
    }
}

void TCLArgs::append_element(const clString& e)
{
    if(!have_error){
	if(have_result)
	    string+=' ';
	string+=e;
	have_result=1;
    }
}

clString TCLArgs::make_list(const clString& item1, const clString& item2)
{
    char* argv[2];
    argv[0]=item1();
    argv[1]=item2();
    char* list=Tcl_Merge(2, argv);
    clString res(list);
    free(list);
    return res;
}

clString TCLArgs::make_list(const clString& item1, const clString& item2,
			const clString& item3)
{
    char* argv[3];
    argv[0]=item1();
    argv[1]=item2();
    argv[2]=item3();
    char* list=Tcl_Merge(3, argv);
    clString res(list);
    free(list);
    return res;
}

clString TCLArgs::make_list(const Array1<clString>& items)
{
    char** argv=scinew char*[items.size()];
    for(int i=0;i<items.size();i++)
	argv[i]=items[i]();
    char* list=Tcl_Merge(items.size(), argv);
    clString res(list);
    free(list);
    delete[] argv;
    return res;
}

void TCL::reset_vars()
{
    for(int i=0;i<vars.size();i++)
	vars[i]->reset();
}

void TCL::register_var(TCLvar* v)
{
    vars.add(v);
}

void TCL::unregister_var(TCLvar* v)
{
    for(int i=0;i<vars.size();i++){
	if(vars[i]==v){
	    vars.remove(i);
	    return;
	}
    }
}

int TCL::get_tcl_stringvar(const clString& base, const clString& name,
			   clString& value)
{
    clString n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, n(), TCL_GLOBAL_ONLY);
    if(!l){
	TCLTask::unlock();
	return 0;
    }
    value=l;
    TCLTask::unlock();
    return 1;
}

int TCL::get_tcl_boolvar(const clString& base, const clString& name,
			 int& value)
{
    clString n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, n(), TCL_GLOBAL_ONLY);
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

int TCL::get_tcl_doublevar(const clString& base, const clString& name,
			   double& value)
{
    clString n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, n(), TCL_GLOBAL_ONLY);
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

int TCL::get_tcl_intvar(const clString& base, const clString& name,
			 int& value)
{
    clString n(base+"-"+name);
    TCLTask::lock();
    char* l=Tcl_GetVar(the_interp, n(), TCL_GLOBAL_ONLY);
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

void TCL::set_tclvar(const clString& base, const clString& name,
		     const clString& value)
{
    clString n(base+"-"+name);
    TCLTask::lock();
    Tcl_SetVar(the_interp, n(), value(), TCL_GLOBAL_ONLY);
    TCLTask::unlock();
}
