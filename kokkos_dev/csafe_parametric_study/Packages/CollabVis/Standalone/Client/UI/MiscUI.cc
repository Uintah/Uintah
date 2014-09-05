#include <UI/MiscUI.h>
#include <UI/SVCallback.h>
#include <UI/GuiArgs.h>
#include <UI/OpenGL.h>
#include <UI/UserInterface.h>

#include <Rendering/Renderer.h>
#include <Logging/Log.h>

#include <string>
#include <tcl.h>

namespace SemotusVisum {

struct TCLCommandData {
  SVCallback* object;
  void* userdata;
};

char *
ccast_unsafe(const string &str)
{
  char *result = const_cast<char *>(str.c_str());
  return result;
}

void
execute(const string& str)
{
  int code = Tcl_Eval(UserInterface::getInterp(), ccast_unsafe(str));
  if(code != TCL_OK)
    Tk_BackgroundError(UserInterface::getInterp());
}

int
eval(const string& str, string& result)
{
  int code = Tcl_Eval(UserInterface::getInterp(), ccast_unsafe(str));
  if(code != TCL_OK){
    Tk_BackgroundError(UserInterface::getInterp());
    result="";
  } else {
    result=string(UserInterface::getInterp()->result);
  }
  return code == TCL_OK;
}

void
source_once(const string& filename)
{
  string result;
  if(!eval("source " + filename, result)) {
    char* msg = ccast_unsafe("Couldn't source file '" + filename + "'");
    Tcl_AddErrorInfo(UserInterface::getInterp(),msg);
    Tk_BackgroundError(UserInterface::getInterp());
  }
}

int
do_command(ClientData cd, Tcl_Interp*,
	   int argc, char* argv[])
{
  TCLCommandData* td=(TCLCommandData*)cd;
  GuiArgs args(argc, argv);
  td->object->tcl_command(args, td->userdata);
  if(args.have_result_) {
    Tcl_SetResult(UserInterface::getInterp(),
		  strdup(args.string_.c_str()),
		  (Tcl_FreeProc*)free);
  }
  return args.have_error_?TCL_ERROR:TCL_OK;
}


void
add_command(const string&command, SVCallback* callback,
	    void* userdata)
{
  TCLCommandData* command_data=new TCLCommandData;
  command_data->object=callback;
  command_data->userdata=userdata;
  Tcl_CreateCommand(UserInterface::getInterp(), ccast_unsafe(command),
		    &do_command, command_data, 0);
}

bool
get(const std::string& name, std::string& value)
{
  char* l=Tcl_GetVar(UserInterface::getInterp(), ccast_unsafe(name),
		     TCL_GLOBAL_ONLY);
  if(!l){
    value="";
    return false;
  }
  value=l;
  return true;
}

void
set(const std::string& name, const std::string& value)
{
  Tcl_SetVar(UserInterface::getInterp(), ccast_unsafe(name),
	     ccast_unsafe(value), TCL_GLOBAL_ONLY);
}



void
redraw() {
  Log::log( ENTER, "[redraw] entered, thread id = " + mkString( (int) pthread_self() ) );
  UserInterface::lock();
  draw();
  execute("update idletasks");
  UserInterface::unlock();
  Log::log( LEAVE, "[redraw] leaving" );
}

void
draw() {

  Log::log( ENTER, "[draw] entered, thread id = " + mkString( (int) pthread_self() ) );
  OpenGL::clearscreen();
  //if ( (UserInterface::renderer() != NULL) && (UserInterface::renderer()->view_set == true) )
    if ( UserInterface::renderer() != NULL )
    UserInterface::renderer()->render( false );
  OpenGL::finish();
  Log::log( LEAVE, "[draw] leaving" );
}


}
