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
 *  ThreadStats.cc: Interface to memory stats...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/ThreadStats.h>

namespace SCIRun {


ThreadStats::ThreadStats()
{
    //oldinfo=info=0;
}

ThreadStats::~ThreadStats()
{
}

void ThreadStats::init_tcl()
{
    TCL::add_command("threadstats", this, 0);
}

void ThreadStats::tcl_command(TCLArgs& args, void*)
{
    if(args.count() < 2){
	args.error("threadstats needs a minor command");
	return;
    }
#if 0
    if(args[1] == "ntasks"){
	if(oldinfo){
	    delete oldinfo;
	}
	oldinfo=info;
	info=Task::get_taskinfo();
	args.result(to_string(info->ntasks));
	args.result("0");
    } else if(args[1] == "dbx"){
	int which;
	if(args.count() < 3 || !args[2].get_int(which)){
	    args.error("Bad argument for threadstats dbx");
	    return;
	}
	Task::debug(info->tinfo[which].taskid);
    } else if(args[1] == "changed"){
	if(oldinfo){
	    int i;
	    for(i=0;i<info->ntasks;i++){
		if(i >= oldinfo->ntasks
		   || info->tinfo[i].name != oldinfo->tinfo[i].name
		   || info->tinfo[i].pid != oldinfo->tinfo[i].pid
		   || info->tinfo[i].stacksize != oldinfo->tinfo[i].stacksize
		   || info->tinfo[i].stackused != oldinfo->tinfo[i].stackused){
		    args.append_element(to_string(i));
		}
	    }
	    for(;i<oldinfo->ntasks;i++){
		args.append_element(to_string(i));
	    }
	} else {
	    for(int i=0;i<info->ntasks;i++){
		args.append_element(to_string(i));
	    }
	}
    } else if(args[1] == "thread"){
	int which;
	if(args.count() < 3 || !args[2].get_int(which)){
	    args.error("Bad argument for threadstats dbx");
	    return;
	}
	char buf[200];
	sprintf(buf, "%d|%d|%s (pid %d)",
		info->tinfo[which].stackused/1024,
		info->tinfo[which].stacksize/1024,
		info->tinfo[which].name,
		info->tinfo[which].pid);
	args.result(buf);
    } else {
#endif
	args.error("Unknown minor command for threadstats");
#if 0
    }
#endif
}

} // End namespace SCIRun

