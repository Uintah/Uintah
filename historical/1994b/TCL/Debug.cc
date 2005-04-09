
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

#include <TCL/Debug.h>
#include <string.h>

typedef AVLTree<clString,Array1<clString>*> Debug;

static Debug* debugs=0;
static DebugInfo* debugsiter=0;

DebugSwitch::DebugSwitch(const clString& module, const clString& var)
: flag(module+"_"+var, "Debug", 0)
{
   if(debugs==0){
      debugs=new Debug;
      debugsiter=new DebugInfo(debugs);
   }
   
   if(debugsiter->search(module)){
      debugsiter->get_data()->add(var);
      return;
   }

   Array1<clString>* arr=new Array1<clString>;
   arr->add(var);
   debugs->insert(module, arr);

   // Init flag
   char* env=getenv("SR_DEBUG");
   if((env!=0)&&(strstr(env,(module+"("+var+")")())!=0))
      flag.set(1);
   else
      flag.set(0);
}

DebugSwitch::~DebugSwitch()
{
   if (debugs!=0){
      for(debugsiter->first();debugsiter->ok();++debugsiter){
	 delete debugsiter->get_data();
      }
      delete debugs;
      delete debugsiter;
      debugs=0;
   }
}

DebugInfo* DebugSwitch::get_debuginfo(int &size)
{
   size=debugs->size();
   return debugsiter;
}

