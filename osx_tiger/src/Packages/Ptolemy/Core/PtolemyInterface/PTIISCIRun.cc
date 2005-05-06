// copyright...
//By Ayla and Oscar

#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
//#include <Packages/Ptolemy/Core/PtolemyInterface/PTIIData.h>

#include <main/sci_version.h>

#include <Dataflow/Modules/Render/Viewer.h>
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
	
	///////////////////////
	//***********Maybe a bad place for this
	JNIUtils::modName = readerName;
	
	//string command = "loadnet {/scratch/SCIRun/test.net}";
	if (! netName.empty()) {
		gui->eval("loadnet {" + netName + "}");
		if(! dataPath.empty() && ! readerName.empty()){
			
			mod = net->get_module_by_id(readerName); //example: SCIRun_DataIO_FieldReader_0
			
			GuiInterface* modGui = mod->getGui();
			
			//for testing
			//std::string result;
			//modGui->get("::SCIRun_DataIO_FieldReader_0-filename", result);
			//std::cerr << "result: " << result << std::endl;
			
			// example" modGui->set("::SCIRun_DataIO_FieldReader_0-filename", "/scratch/DATA1.22.0/utahtorso/utahtorso-voltage.tvd.fld");
			std::string state;
			modGui->get("::" + readerName + "-filename", state);
			
			std::cerr << "file name was: " << state << std::endl;
			
			modGui->set("::" + readerName + "-filename", dataPath);
		}
		else if (! readerName.empty()) {
		
			//for running a module that doesnt neccesarily have a file to load
			mod=net->get_module_by_id(readerName); //example: SCIRun_DataIO_FieldReader_0
		}
	}
    // Now activate the TCL event loop
    tcl_task->release_mainloop();

	
	//should just have a general run network here.
	if(runNet == 1 && ! readerName.empty()){
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

Semaphore& Iterate::iterSem()
{
    static Semaphore sem_("iterate ptolemy request semaphore", 0);
    return sem_;
}


//Note that if you add this callback at the beginning of a task
//it is necessary to remove it at the end so it wont modify the static
//semephore in the future when it is needed again.
void Iterate::iter_callback(void *data)
{
	iterSem().up();
}

std::string Iterate::returnValue = "OK";

void Iterate::run()
{
	
	JNIUtils::sem().down();
	
	string name;
	
	//get a pointer to the viewer if we need it and check to see if its valid
	Viewer* viewer;
	if(picPath != ""){
		viewer = (Viewer*)JNIUtils::cachedNet->get_module_by_id("SCIRun_Render_Viewer_0");
		if(viewer == 0){
			returnValue = "no viewer present";
			JNIUtils::sem().up();
			return;
		}
	}
	
	Scheduler* sched = JNIUtils::cachedNet->get_scheduler();
	sched->add_callback(iter_callback, this);
	
	//set the initial parameters
	Module* modptr;
	GuiInterface* modGui;

	for(jint i = 0; i < size1; i++){
		modptr = JNIUtils::cachedNet->get_module_by_id(doOnce[i]);
		if(modptr == 0){
			returnValue = doOnce[i] + " not present in the network";
			sched->remove_callback(iter_callback, this);
			JNIUtils::sem().up();
			return;
		}
		i++;
		modGui = modptr->getGui();
		ASSERT(modGui);
		
		//std::cout << "doOnce " << doOnce[i-1] << " " << doOnce[i] << " " << doOnce[i+1] << std::endl;
		
		modGui->set("::" + doOnce[i-1] + doOnce[i], doOnce[i+1]);
		i++;
	
	
	}
	
	
	//iterate through the tasks given to SCIRun
	for(jint i = 0; i < numParams; i++){
		
		for(jint j = 0; j < size2; j=j+numParams-i){
			//TODO ask if it would be better here to have a dynamically
			//allocated array of module pointers for each thing
			//depends on how efficient getmodbyid really is
			modptr = JNIUtils::cachedNet->get_module_by_id(iterate[j]);
			if(modptr == 0){
				returnValue = iterate[j] + " not present in the network";
				sched->remove_callback(iter_callback, this);
				JNIUtils::sem().up();
				return;
			}
			j++;
			modGui = modptr->getGui();
			ASSERT(modGui);
		
			//std::cout << "iterate " << iterate[j-1] << " " << iterate[j] << " " << iterate[j+i+1] << std::endl;
			
			modGui->set("::" + iterate[j-1] + iterate[j], iterate[j+i+1]);
			j=j+i+1;
		}
		
		//execute all and wait for it to finish
		gui->eval("updateRunDateAndTime {0}");
		gui->eval("netedit scheduleall");		
		iterSem().down();
		
		//if you want to save the picture
		//TODO do we care if there is no viewer that is getting the messages?
		//TODO worry about saving over existing images.  would be cool to prompt
		// the user if they are going to save over an image that exists already
		if(picPath != ""){
			name = picPath + "image" + to_string(i) + ".ppm";

			//when the viewer is done save the image
			ViewerMessage *msg1 = scinew ViewerMessage
			(MessageTypes::ViewWindowDumpImage,"::SCIRun_Render_Viewer_0-ViewWindow_0",name, "ppm","640","480");
			viewer->mailbox.send(msg1); 

			ViewerMessage *msg2 = scinew ViewerMessage("::SCIRun_Render_Viewer_0-ViewWindow_0");
			viewer->mailbox.send(msg2);
			
			std::cout << "sending message: " << i << std::endl; 
		}//else we do not try and save pictures
		
	}
	Iterate::returnValue = "OK";
	sched->remove_callback(iter_callback, this);	
	JNIUtils::sem().up();
}

Iterate::~Iterate(){
	//free dynamically allocated memory
	delete [] doOnce;
	delete [] iterate;
}

void SignalExecuteReady::run()
{
    //converterMod->sendJNIData(np, nc, pDim, cDim, *p, *c);
    JNIUtils::dataSem().up();
}

void QuitSCIRun::run()
{
    // what else for shutdown?
    Thread::exitAll(0);
}
