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

#include <Classlib/Debug.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <string.h>

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

#ifdef __GNUG__
// Template instantiations
#include <Classlib/Array1.cc>
template class Array1<DebugSwitch*>;

#include <Classlib/AVLTree.cc>
template class AVLTree<clString, DebugVars*>;
template class AVLTreeIter<clString, DebugVars*>;
template class TreeLink<clString, DebugVars*>;

#endif
