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
 *  Debug.cc: Implementation of debug switches
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Util/Debug.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <stdlib.h>
#include <string.h>

namespace SCIRun {

static Debug* debugs=0;
static int initialized=0;

DebugSwitch::DebugSwitch(const string& module, const string& var)
: module(module), var(var), flagvar(0)
{
    if(initialized){
	cerr << "DebugSwitch was added after we were already initialized!";
    }
    if(debugs==0){
	debugs=scinew Debug;
    }

    // Init flag
    // cerr << "This part is broken...\n";
    char* env=getenv("SCI_DEBUG");
    if((env!=0)&&(strstr(env,(module+"("+var+")").c_str())!=0))
	flagvar=1;

    DebugIter debugsiter(debugs);
    if(debugsiter.search(module)){
	DebugSwitch* that=this;
	debugsiter.get_data()->add(that);
	return;
    }

    DebugVars* arr=scinew DebugVars;
    DebugSwitch* that=this;
    arr->add(that);
    debugs->insert(module, arr);

}

DebugSwitch::~DebugSwitch()
{
    if (debugs!=0){
       DebugIter debugsiter(debugs);
       for(debugsiter.first();debugsiter.ok();++debugsiter){
	   delete debugsiter.get_data();
       }
       delete debugs;
       debugs=0;
   }
}

Debug* DebugSwitch::get_debuginfo()
{
    return debugs;
}

string DebugSwitch::get_module()
{
    return module;
}

string DebugSwitch::get_var()
{
    return var;
}

int* DebugSwitch::get_flagpointer()
{
    return &flagvar;
}

ostream& operator<<(ostream& str, DebugSwitch& db)
{
    str << db.module << ":" << db.var << " - ";
    return str;
}

} // End namespace SCIRun

