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

#include <Core/GuiInterface/DebugSettings.h>

#include <Core/Util/Debug.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

namespace SCIRun {


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
		variables.add(scinew GuiVarintp(sw->get_flagpointer(),
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

} // End namespace SCIRun

