//static char *id="@(#) $Id$";

/*
 *  DebugSettings.cc: Interface to debug settings...
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/TclInterface/DebugSettings.h>

#include <SCICore/Util/Debug.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Util::Debug;
using SCICore::Util::DebugSwitch;
using SCICore::Util::DebugIter;
using SCICore::Util::DebugVars;

DebugSettings::DebugSettings()
{
}

DebugSettings::~DebugSettings()
{
}

void DebugSettings::init_tcl()
{
   TCL::add_command("debugsettings", this, 0);
}

void DebugSettings::tcl_command(TCLArgs& args, void*)
{
    if(args.count() > 1){
	args.error("debugsettings needs no minor command");
	return;
    }
    int makevars=0;
    if(variables.size() == 0)
	makevars=1;

    Debug* debug=DebugSwitch::get_debuginfo();
    if(!debug){
	args.result("");
	return;
    }
    Array1<clString> debuglist(debug->size());
    
    DebugIter iter(debug);
    int i;
    for(iter.first(),i=0;iter.ok();++iter,++i){
	DebugVars& debug_vars=*iter.get_data();
	Array1<clString> vars(debug_vars.size());
	for(int j=0;j<debug_vars.size();j++){
	    DebugSwitch* sw=debug_vars[j];
	    vars[j]=sw->get_var();
	    if(makevars)
		variables.add(scinew TCLvarintp(sw->get_flagpointer(),
					     sw->get_module(), sw->get_var(),
					     0));
	}

	// Make a list of the variables
	clString varlist(args.make_list(vars));
	
	// Make a list with the module name and this list
	clString mlist(args.make_list(iter.get_key(), varlist));

	// Put this in the array
	debuglist[i]=mlist;
    }

    args.result(args.make_list(debuglist));
}

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/09/08 02:26:54  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:39:41  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:13  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//
