
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

#include <share/share.h>

#include <Containers/Array1.h>
#include <Containers/AVLTree.h>
#include <Containers/String.h>

namespace SCICore {
namespace Util {

using SCICore::Containers::Array1;
using SCICore::Containers::AVLTree;
using SCICore::Containers::AVLTreeIter;
using SCICore::Containers::clString;

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
class SHARE DebugSwitch {
    clString module;
    clString var;
    friend ostream& operator<<(ostream&, DebugSwitch&);
public:
    // Creation and destruction of debug switches
    DebugSwitch(const clString& module, const clString& var);
    ~DebugSwitch();   
   
    inline int operator!() const;
    inline operator int() const;
   
    static Debug* get_debuginfo();
    int* get_flagpointer();
    clString get_module();
    clString get_var();
private:
    int flagvar;
};


inline int
DebugSwitch::operator!() const
{
//    cerr << "flagvar=" << flagvar << endl;
    return !flagvar;
}

inline
DebugSwitch::operator int() const
{
//    cerr << "flagvar=" << flagvar << endl;
    return flagvar;
}

#define PRINTVAR(sw, varname) \
    if(sw)cerr << sw << ": "#varname"=" << varname << endl

} // End namespace Util
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:35  mcq
// Initial commit
//
// Revision 1.4  1999/07/09 01:18:58  moulding
// added SHARE for win32 shared libraries (.dll's)
//
// Revision 1.3  1999/05/06 19:56:26  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:38  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif
