//static char *id="@(#) $Id$";

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

#include <SCICore/TclInterface/ThreadStats.h>

#include <iostream.h>
#include <stdio.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::to_string;

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

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/28 17:54:52  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/19 23:18:08  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:39:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:17  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
