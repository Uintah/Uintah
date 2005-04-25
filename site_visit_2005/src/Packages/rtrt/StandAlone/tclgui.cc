#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace SCIRun;

class RtrtTclGui: public GuiCallback {
public:
  RtrtTclGui(GuiInterface* gui);

  static RtrtTclGui* startup(int argc, char* argv[], char **environment);
private:
  virtual void	tcl_command(GuiArgs&, void*);

  GuiInterface* tcl_int;
  GuiContext* ctx;
};

RtrtTclGui::RtrtTclGui(GuiInterface* gui):
  tcl_int(gui)
{
  gui->add_command("rtrtgui", this, 0);
  gui->add_command("netedit", this, 0);
  ASSERT(sci_getenv("SCIRUN_SRCDIR"));
  gui->source_once(sci_getenv("SCIRUN_SRCDIR")+
		   string("/Dataflow/GUI/NetworkEditor.tcl"));
  gui->source_once(sci_getenv("SCIRUN_SRCDIR")+
		   string("/Packages/rtrt/Dataflow/GUI/RtrtTclGui.tcl"));

  string ctx_name("::rtrtgui_ctx_01");
  ctx = tcl_int->createContext(ctx_name);
  gui->execute(string("setrtrtctx ") + ctx_name);

  gui->execute("makeRtrtGui");

  GuiInt* nprocs = new GuiInt(ctx->subVar("nprocs"));
  cerr << "RtrtTclGui::RtrtTclGui(): nprocs = "<<nprocs->get()<<"\n";
  nprocs->set(2);
  cerr << "RtrtTclGui::RtrtTclGui(): nprocs = "<<nprocs->get()<<"\n";
  gui->execute(ctx_name + string(" printnprocs"));
  gui->execute(ctx_name + string(" printmodname"));
}

void RtrtTclGui::tcl_command(GuiArgs& args, void*) {
  if(args.count() < 2) {
    args.error("rtrtgui needs a minor command");
    return;
  }
  if(args[1] == "quit"){
    Thread::exitAll(0);
  } else if(args[1] == "hello") {
    cout << "Hello!\n";
  }
  // These need to be in here to placate calls that ask for netedit.
  // Not all the functions supported by netedit are reimplemented,
  // just the ones we care about.
  else if (args[1] == "getenv" && args.count() == 3){
    const char *result = sci_getenv( args[2] );
    if (result) {
      args.result(string(result));
    }
    return;
  } else if (args[1] == "setenv" && args.count() == 4){
    sci_putenv(args[2], args[3]);
  }
}

RtrtTclGui* RtrtTclGui::startup(int argc, char* argv[], char **environment) {
  create_sci_environment(environment, 0);

  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(1, argv);// Only passes program name to TCL
  // We need to start the thread in the NotActivated state, so we can
  // change the stack size.  The 0 is a pointer to a ThreadGroup which
  // will default to the global thread group.
  Thread* t=new Thread(tcl_task,"TCL main event loop",0, Thread::NotActivated);
  t->setStackSize(1024*1024);
  // False here is stating that the tread was stopped or not.  Since
  // we have never started it the parameter should be false.
  t->activate(false);
  t->detach();
  tcl_task->mainloop_waitstart();

  // Create user interface link
  TCLInterface *gui = new TCLInterface();
  RtrtTclGui* rtrtgui = new RtrtTclGui(gui);
  
  // Here we can execute tcl commands via gui->eval("command");
  
  // Now activate the TCL event loop once we started up everthing.
  tcl_task->release_mainloop();

  return rtrtgui;
}

int main(int argc, char* argv[], char **environment) {

  RtrtTclGui* rtrtgui = RtrtTclGui::startup(argc, argv, environment);
  
#ifdef _WIN32
  // windows has a semantic problem with atexit(), so we wait here instead.
  HANDLE forever = CreateSemaphore(0,0,1,"forever");
  WaitForSingleObject(forever,INFINITE);
#endif

#if !defined(__sgi)
  Semaphore wait("main wait", 0);
  wait.down();
#endif

  return 0;
}
