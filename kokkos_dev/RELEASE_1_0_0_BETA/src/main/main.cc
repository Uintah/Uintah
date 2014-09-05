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
 *  uintah.cc: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Modified for distributed Dataflow by:
 *   Michelle Miller
 *   May 1998
 *
 *  Copyright (C) 1999 U of U
 */

#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
//#include <Dataflow/Distributed/SlaveController.h>
#include <Dataflow/Network/PackageDB.h>
#include <Core/GuiInterface/GuiServer.h>
#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Thread.h>
#include <Core/Containers/String.h>

#ifdef SCI_PARALLEL
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#endif

#ifdef _WIN32
#include <afxwin.h>
#endif

using namespace SCIRun;

namespace SCIRun {
extern bool global_remote;
void set_guiManager( GuiManager * );
}

//extern void set_guiManager( GuiManager* );

int global_argc;
char** global_argv;

#ifndef PSECORETCL
#error You must set PSECORETCL to the Dataflow/Tcl path
#endif

#ifndef SCICORETCL
#error You must set SCICORETCL to the Core/Tcl path
#endif

#ifndef DEFAULT_PACKAGE_PATH
#error You must set a DEFAULT_PACKAGE_PATH or life is pretty dull
#endif

#ifndef ITCL_WIDGETS
#error You must set ITCL_WIDGETS to the iwidgets/scripts path
#endif

// master creates slave by rsh "sr -slave hostname portnumber"
int main(int argc, char** argv)
{
  global_argc=argc;
  global_argv=argv;

#ifdef SCI_PARALLEL
  try {
    PIDL::PIDL::initialize(argc, argv);
  } catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  } 
#endif

  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(argc, argv);
  Thread* t=new Thread(tcl_task, "TCL main event loop");
  t->detach();
  tcl_task->mainloop_waitstart();

  // Set up the TCL environment to find core components
  clString result;
  TCL::eval("global PSECoreTCL CoreTCL",result);
  TCL::eval("set DataflowTCL "PSECORETCL,result);
  TCL::eval("set CoreTCL "SCICORETCL,result);
  TCL::eval("lappend auto_path "SCICORETCL,result);
  TCL::eval("lappend auto_path "PSECORETCL,result);
  TCL::eval("lappend auto_path "ITCL_WIDGETS,result);

  // Create initial network
  // We build the Network with a 1, indicating that this is the
  // first instantiation of the network class, and this network
  // should read the command line specified files (if any)
  Network* net=new Network(1);

  // Fork off task for the network editor.  It is a detached
  // task, and the Task* will be deleted by the task manager
  NetworkEditor* gui_task=new NetworkEditor(net);

  // Activate the network editor and scheduler.  Arguments and return
  // values are meaningless
  Thread* t2=new Thread(gui_task, "Scheduler");
  t2->setDaemon(true);
  t2->detach();

  // wait for the main window to display before continuing the startup.
  TCL::eval("tkwait visibility .top.globalViewFrame.canvas",result);

  // Load in the default packages
  if(getenv("PACKAGE_PATH")!=NULL)
    packageDB.loadPackage(getenv("PACKAGE_PATH"));
  else
    packageDB.loadPackage(DEFAULT_PACKAGE_PATH);
    

#if 0
  // startup master-side GuiServer, even when no slave..
  // XXX: Is this a remnant of dist'd stuff?  Michelle?
  GuiServer* gui_server = new GuiServer;
  Thread* t3=new Thread(gui_server, "GUI server thread");
  t3->setDaemon(true);
  t3->detach();
#endif

  // Now activate the TCL event loop
  tcl_task->release_mainloop();

#ifdef _WIN32
  // windows has a semantic problem with atexit(), so we wait here instead.
  HANDLE forever = CreateSemaphore(0,0,1,"forever");
  WaitForSingleObject(forever,INFINITE);
#endif

#ifndef __sgi
  Semaphore wait("main wait", 0);
  wait.down();
#endif
	
  // Never reached
  return 0;

}
