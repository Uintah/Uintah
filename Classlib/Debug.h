
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

#include <Classlib/Array1.h>
#include <Classlib/AVLTree.h>
#include <Classlib/String.h>

class DebugSwitch;
typedef Array1<DebugSwitch*> DebugVars;
typedef AVLTree<clString, DebugVars*> Debug;
typedef AVLTreeIter<clString, DebugVars*> DebugIter;

/*
 * DebugSwitchs initialize to no debugging unless the environment
 * variable SCI_DEBUG contains an entry of the form module(var),
 * where module and var are identical to the module and var
 * parameters, respectivly.
 */

// DebugSwitch class.
class DebugSwitch {
    clString module;
    clString var;
    friend ostream& operator<<(ostream&, DebugSwitch&);
public:
    // Creation and destruction of debug switches
    DebugSwitch(const clString& module, const clString& var);
    ~DebugSwitch();   
   
    inline int operator!();
   
    static Debug* get_debuginfo();
    int* get_flagpointer();
    clString get_module();
    clString get_var();
private:
    int flagvar;
};


inline int
DebugSwitch::operator!()
{
    return !flagvar;
}

#define PRINTVAR(sw, varname) \
    cerr << sw << ": "#varname"=" << varname << endl

#endif

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

#include <Classlib/Array1.h>
#include <Classlib/AVLTree.h>
#include <Classlib/String.h>

class DebugSwitch;
typedef Array1<DebugSwitch*> DebugVars;
typedef AVLTree<clString, DebugVars*> Debug;
typedef AVLTreeIter<clString, DebugVars*> DebugIter;

/*
 * DebugSwitchs initialize to no debugging unless the environment
 * variable SCI_DEBUG contains an entry of the form module(var),
 * where module and var are identical to the module and var
 * parameters, respectivly.
 */

// DebugSwitch class.
class DebugSwitch {
    clString module;
    clString var;
    friend ostream& operator<<(ostream&, DebugSwitch&);
public:
    // Creation and destruction of debug switches
    DebugSwitch(const clString& module, const clString& var);
    ~DebugSwitch();   
   
    inline int operator!();
   
    static Debug* get_debuginfo();
    int* get_flagpointer();
    clString get_module();
    clString get_var();
private:
    int flagvar;
};


inline int
DebugSwitch::operator!()
{
	cerr << "flagvar=" << flagvar << endl;
    return !flagvar;
}

#define PRINTVAR(sw, varname) \
    if(!!sw)cerr << sw << ": "#varname"=" << varname << endl

#endif
