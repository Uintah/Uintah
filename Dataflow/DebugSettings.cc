
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

#include <Dataflow/DebugSettings.h>

#include <Classlib/Debug.h>
#include <TCL/TCLvar.h>

#include <iostream.h>
#include <string.h>
#include <stdio.h>

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
		variables.add(new TCLvarintp(sw->get_flagpointer(),
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
