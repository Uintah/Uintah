
/*
 *  DebugSettings.cc: Interface to debug settings...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/DebugSettings.h>

#include <Classlib/NotFinished.h>
#include <TCL/Debug.h>

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

   int size=0;
   DebugInfo* dinfo=DebugSwitch::get_debuginfo(size);
   cout << "Size: " << size << endl;
   Array1<clString> debugs(size);

   args.result("!");
   /*
   int i;
   for(i=0,dinfo->first();dinfo->ok();i++,++(*dinfo)){
      Array1<clString>* debug(dinfo->get_data());
      debug.add(dinfo->get_key());
      cout << i << " " << dinfo->get_key() << endl;
      debugs[i]=args.make_list(*debug);
   }

   args.result(args.make_list(debugs));*/
}
