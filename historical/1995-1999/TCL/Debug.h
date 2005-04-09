
/*
 *  Debug.h: Interface to the debug switches.
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Debug_h
#define SCI_project_Debug_h 1

#include <TCL/TCLvar.h>
#include <Classlib/Array1.h>
#include <Classlib/AVLTree.h>
#include <Classlib/String.h>

typedef AVLTreeIter<clString,Array1<clString>*> DebugInfo;

/*
 * DebugSwitchs initialize to no debugging unless the environment
 * variable SR_DEBUG contains an entry of the form module(var),
 * where module and var are identical to the module and var
 * parameters, respectivly.
 */

// DebugSwitch class.
class DebugSwitch {
public:
   // Creation and destruction of debug switches
   DebugSwitch(const clString& module, const clString& var);
   ~DebugSwitch();   
   
   inline int operator!();
   
   static DebugInfo* get_debuginfo(int &size);
   
private:
   TCLvarint flag;
};


inline int
DebugSwitch::operator!()
{
    return !flag.get();
}

#endif
