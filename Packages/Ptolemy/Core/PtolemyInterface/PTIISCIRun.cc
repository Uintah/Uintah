// copyright...

#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <main/sci_version.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Environment.h>
#include <Core/Util/sci_system.h>
#include <Core/Comm/StringSocket.h>
#include <Core/Thread/Thread.h>

#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>

#include <Core/Services/ServiceLog.h>
#include <Core/Services/ServiceDB.h>
#include <Core/Services/ServiceManager.h>
#include <Core/SystemCall/SystemCallManager.h>

#include <sys/stat.h>
#include <fcntl.h>

#include <string>
#include <iostream>
#include <unistd.h>


//using namespace SCIRun;

static TCLInterface *gui = 0;


//////////////////////////////////////////////////////////////////////////
//  copied verbatim from main.cc
//////////////////////////////////////////////////////////////////////////
// show_licence_and_copy_sciunrc is not in Core/Util/Environment.h because it
// depends on GuiInterface to present the user with the license dialog.
void
show_license_and_copy_scirunrc(GuiInterface *gui) {
  const string tclresult = gui->eval("licenseDialog 1");
  if (tclresult == "cancel")
  {
    Thread::exitAll(1);
  }
  // check to make sure home directory is there
  const char* HOME = sci_getenv("HOME");
  const char* srcdir = sci_getenv("SCIRUN_SRCDIR");
  ASSERT(HOME);
  ASSERT(srcdir);
  if (!HOME) return;
  // If the user accepted the license then create a .scirunrc for them
  if (tclresult == "accept") {
    string homerc = string(HOME)+"/.scirunrc";
    string cmd;
    if (gui->eval("validFile "+homerc) == "1") {
      string backuprc = homerc+"."+string(SCIRUN_VERSION)+
	string(SCIRUN_RCFILE_SUBVERSION);
      cmd = string("cp -f ")+homerc+" "+backuprc;
      std::cout << "Backing up " << homerc << " to " << backuprc << std::endl;
      if (sci_system(cmd.c_str())) {
	std::cerr << "Error executing: " << cmd << std::endl;
      }
    }

    cmd = string("cp -f ")+srcdir+string("/scirunrc ")+homerc;
    std::cout << "Copying " << srcdir << "/scirunrc to " <<
      homerc << "...\n";
    if (sci_system(cmd.c_str())) {
      std::cerr << "Error executing: " << cmd << std::endl;
    } else { 
      // if the scirunrc file was copied, then parse it
      parse_scirunrc(homerc);
    }
  }
}


//////////////////////////////////////////////////////////////////////////
//  mostly copied from main.cc
void StartSCIRun::run()
{
    std::cerr << "StartSCIRun::run()" << std::endl;

    char* argv[2];
    argv[0] = "sr";
    argv[1] = 0;

    // Setup the SCIRun key/value environment
    create_sci_environment(0, 0);
    sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);

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
    gui = new TCLInterface();

    // Create initial network
    packageDB = new PackageDB(gui);
    //packageDB = new PackageDB(0);
    Network* net=new Network();
    JNIUtils::cachedNet = net;

    Scheduler* sched_task=new Scheduler(net);
    new NetworkEditor(net, gui);

    //gui->execute("wm withdraw ."); // used by SCIRun2 Dataflow Component Model
    //packageDB->setGui(gui);

    // If the user doesnt have a .scirunrc file, provide them with a default one
    if (!find_and_parse_scirunrc()) show_license_and_copy_scirunrc(gui);

    // Activate the scheduler.  Arguments and return values are meaningless
    Thread* t2=new Thread(sched_task, "Scheduler");
    t2->setDaemon(true);
    t2->detach();

    gui->eval("set PowerApp 0");
    // wait for the main window to display before continuing the startup.
    gui->eval("wm deiconify .");
    gui->eval("tkwait visibility $minicanvas");
    gui->eval("showProgress 1 0 1");

    packageDB->loadPackage();  // load the packages
    gui->eval("hideProgress");

	// Check the dynamic compilation directory for validity
	sci_putenv("SCIRUN_ON_THE_FLY_LIBS_DIR",gui->eval("getOnTheFlyLibsDir"));

	
    // Activate "File" menu sub-menus once packages are all loaded.
    gui->eval("activate_file_submenus");
	
	
	
	Module* mod;
	//string command = "loadnet {/scratch/SCIRun/test.net}";
	if (runNet != 0 && netName != "") {
		gui->eval("loadnet {" + netName + "}");
		if(dataPath != "" && readerName != ""){
			
			mod=net->get_module_by_id(readerName); //example: SCIRun_DataIO_FieldReader_0
			
			GuiInterface* modGui = mod->getGui();
			
			//for testing
			//std::string result;
			//modGui->get("::SCIRun_DataIO_FieldReader_0-filename", result);
			//std::cerr << "result: " << result << std::endl;
			
			// example" modGui->set("::SCIRun_DataIO_FieldReader_0-filename", "/scratch/DATA1.22.0/utahtorso/utahtorso-voltage.tvd.fld");
			modGui->set("::" + readerName + "-filename", dataPath);
		}
		else if(readerName != ""){
			//for running a module that doesnt neccesarily have a file to load
			mod=net->get_module_by_id(readerName); //example: SCIRun_DataIO_FieldReader_0
		}
		
	}
    // Now activate the TCL event loop
    tcl_task->release_mainloop();

	
	//should just have a general run network here.
	if(runNet == 1 && readerName != ""){
		mod->want_to_execute();  //tell the first module that it wants to execute
	}
	
    JNIUtils::sem().up();

#if 0
//#ifdef _WIN32
// windows has a semantic problem with atexit(), so we wait here instead.
//  HANDLE forever = CreateSemaphore(0,0,1,"forever");
//  WaitForSingleObject(forever,INFINITE);
//#endif
//
//#if !defined(__sgi)
//  Semaphore wait("main wait", 0);
//  wait.down();
//#endif
#endif
}


void AddModule::run()
{
    JNIUtils::sem().down();
    std::cerr << "AddModule::run: " << command << std::endl;
	
    JNIUtils::sem().up();
}

void SignalExecuteReady::run()
{
    //converterMod->sendJNIData(np, nc, pDim, cDim, *p, *c);
    JNIUtils::dataSem().up();
}
