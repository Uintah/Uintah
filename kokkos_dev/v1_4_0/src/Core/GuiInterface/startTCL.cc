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
 *  startTCL.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Thread.h>
#include <sci_defs.h>
#include <string>
using namespace std;
using namespace SCIRun;

extern "C" void start_TCL();

#define PSECORETCL SCIRUN_SRCDIR "/Dataflow/GUI"
#define SCICORETCL SCIRUN_SRCDIR "/Core/GUI"
#define ITCL_WIDGETS "/home/sparker/SCIRun/SCIRun_Thirdparty_32_linux/lib/iwidgets/scripts"

void start_TCL()
{
  int argc=1;
  char* argv[2];
  argv[0]="sr";
  argv[1]=0;
  // Start up TCL...
  TCLTask* tcl_task = new TCLTask(argc, argv);
  Thread* t=new Thread(tcl_task, "TCL main event loop");
  t->detach();
  tcl_task->mainloop_waitstart();

  // Set up the TCL environment to find core components
  TCL::execute("global PSECoreTCL CoreTCL");
  TCL::execute("set DataflowTCL "PSECORETCL);
  TCL::execute("set CoreTCL "SCICORETCL);
  TCL::execute("lappend auto_path "SCICORETCL);
  TCL::execute("lappend auto_path "PSECORETCL);
  TCL::execute("lappend auto_path "ITCL_WIDGETS);

  TCL::execute("wm withdraw .");

  // Now activate the TCL event loop
  tcl_task->release_mainloop();
}
