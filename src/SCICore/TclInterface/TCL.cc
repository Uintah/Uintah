
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

#include <SCICore/Containers/String.h>
//#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/Task.h>
#include <SCICore/TclInterface/GuiManager.h>
#include <SCICore/TclInterface/Remote.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <tcl.h>
#include <tk.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>                             // defines read() and write()
#endif

extern "C" Tcl_Interp* the_interp;

namespace SCICore {
namespace TclInterface {

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
static clString prefix()
{
    static int haveit=0;
    static clString pf;
    if(!haveit){
	char* p = getenv("PSE_WORK");
	if(p){
	    pf=clString(p);
	} else {
		printf("Error: PSE_WORK variable not set!\n");
	    Task::exit_all(-1);
	}
    }
    return pf;
}

static clString application()
{
    static int haveit=0;
    static clString app;
    if(!haveit){
	char* a = getenv("APPLICATION");
	if(a){
	    app=clString(a);
	} else {
		printf("Error; APPLICATION environment variable not set!\n");
	    Task::exit_all(-1);
	}
    }
    return app;
}

#endif

void TCL::execute(const clString& string)
{
#ifndef _WIN32
    if (gm != NULL) {
    	int skt = gm->getConnection();

	printf ("TCL::execute(%s): Got skt from gm->getConnection() = %d", 
	string(), skt);

	// format request - no TCL variable name, just a string to execute
	TCLMessage msg;
	msg.f = exec;
	strcpy (msg.un.tstring, string());

	// send request to server - no need for reply, error goes to Tk
	if (sendRequest (&msg, skt) == -1) {
	    // error case ???
	}
        gm->putConnection (skt);
    }
    else {
        TCLTask::lock();
        int code = Tcl_Eval(the_interp, const_cast<char *>(string()));
        if(code != TCL_OK)
	    Tk_BackgroundError(the_interp);
        TCLTask::unlock();
    }
}

void TCL::execute(char* string)
{
    if (gm != NULL) {
    	int skt = gm->getConnection();
printf ("TCL::execute(%s): Got skt from gm->getConnection() = %d", string,skt);

	// format request - no TCL variable name, just a string to execute
	TCLMessage msg;
	msg.f = exec;
	strcpy (msg.un.tstring, string);

	// send request to server - no need for reply, error goes to Tk
	if (sendRequest (&msg, skt) == -1) {
	    // error case ???
	}
        gm->putConnection (skt);
    }
    else 
#endif
	{
	    //printf("TCL::execute() 1\n");
	    TCLTask::lock();
        int code = Tcl_Eval(the_interp, string);
        if(code != TCL_OK)
		{
			Tk_BackgroundError(the_interp);
			printf("Tcl_Eval(the_inter,%s) failed\n",string);
		}
        TCLTask::unlock();
    }
}

int TCL::eval(const clString& string, clString& result)
{
    TCLTask::lock();
    int code = Tcl_Eval(the_interp, const_cast<char *>(string()));
    if(code != TCL_OK){
	Tk_BackgroundError(the_interp);
	result="";
    } else {
	result=clString(the_interp->result);
    }
    TCLTask::unlock();
    return code == TCL_OK;
}

int TCL::eval(char* str, clString& result)
{
    TCLTask::lock();
    int code = Tcl_Eval(the_interp, str);
    if(code != TCL_OK){
	Tk_BackgroundError(the_interp);
	result="";
    } else {
	result=clString(the_interp->result);
    }
    TCLTask::unlock();
    return code != TCL_OK;
}

void TCL::source_once(const clString& filename)
{
  clString result;
  if(!eval((clString("source ")+filename)(),result)) {
    char* msg=const_cast<char *>
             ((clString("Couldn't source file '")+filename+"'")());
    Tcl_AddErrorInfo(the_interp,msg);
    Tk_BackgroundError(the_interp);
  }

#if 0
    int code;

    TCLTask::lock();

    clString pse_filename(filename);

    char* fn=const_cast<char *>(pse_filename());
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
    Tcl_CreateCommand(the_interp, const_cast<char *>(command()), do_command, command_data, 0);
    TCLTask::unlock();
}

void
TCL::delete_command( const clString& command )
{
    TCLTask::lock();
    Tcl_DeleteCommand(the_interp, const_cast<char *>(command()) );
    TCLTask::unlock();
}

TCL::TCL()
{
}

TCL::~TCL()
{
}

void TCL::emit_vars(ostream& out)
{
    for(int i=0;i<vars.size();i++)
      {
	cerr << "emit: " << vars[i]->str() << endl;
        vars[i]->emit(out);
      }
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
    argv[0]=const_cast<char *>(item1());
    argv[1]=const_cast<char *>(item2());
    char* list=Tcl_Merge(2, argv);
    clString res(list);
    free(list);
    return res;
}

clString TCLArgs::make_list(const clString& item1, const clString& item2,
			const clString& item3)
{
    char* argv[3];
    argv[0]=const_cast<char *>(item1());
    argv[1]=const_cast<char *>(item2());
    argv[2]=const_cast<char *>(item3());
    char* list=Tcl_Merge(3, argv);
    clString res(list);
    free(list);
    return res;
}

clString TCLArgs::make_list(const Array1<clString>& items)
{
    char** argv=scinew char*[items.size()];
    for(int i=0;i<items.size();i++)
	argv[i]=const_cast<char *>(items[i]());
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
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(n()), TCL_GLOBAL_ONLY);
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
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(n()), TCL_GLOBAL_ONLY);
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
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(n()), TCL_GLOBAL_ONLY);
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
    char* l=Tcl_GetVar(the_interp, const_cast<char *>(n()), TCL_GLOBAL_ONLY);
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
    Tcl_SetVar(the_interp, const_cast<char *>(n()), const_cast<char *>(value()), TCL_GLOBAL_ONLY);
    TCLTask::unlock();
}

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/08/23 20:11:51  sparker
// GenAxes had no UI
// Removed extraneous print statements
// Miscellaneous compilation issues
// Fixed an authorship error
//
// Revision 1.4  1999/08/19 23:18:07  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:21  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:39:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:15  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:11:03  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/05/17 21:55:35  dav
// added new Makefiles and Makefile.main to src tree
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
