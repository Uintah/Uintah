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

#include <SCICore/Util/Debug.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>

namespace SCICore {
namespace Util {

static Debug* debugs=0;
static int initialized=0;

DebugSwitch::DebugSwitch(const clString& module, const clString& var)
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
    if((env!=0)&&(strstr(env,(module+"("+var+")")())!=0))
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

clString DebugSwitch::get_module()
{
    return module;
}

clString DebugSwitch::get_var()
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

} // End namespace Util
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/09/04 06:01:56  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.2  1999/08/17 06:39:55  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:35  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//
