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

#include <Core/share/share.h>

#include <Core/Containers/Array1.h>
#include <Core/Containers/AVLTree.h>

namespace SCIRun {


class DebugSwitch;
typedef Array1<DebugSwitch*> DebugVars;
typedef AVLTree<string, DebugVars*> Debug;
typedef AVLTreeIter<string, DebugVars*> DebugIter;

/*
 * DebugSwitchs initialize to no debugging unless the environment
 * variable SCI_DEBUG contains an entry of the form module(var),
 * where module and var are identical to the module and var
 * parameters, respectivly.
 */

// DebugSwitch class.
class SCICORESHARE DebugSwitch {
    string module;
    string var;
    friend std::ostream& operator<<(std::ostream&, DebugSwitch&);
public:
    // Creation and destruction of debug switches
    DebugSwitch(const string& module, const string& var);
    ~DebugSwitch();   
   
    inline int operator!() const;
    inline operator int() const;
   
    static Debug* get_debuginfo();
    int* get_flagpointer();
    string get_module();
    string get_var();
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

} // End namespace SCIRun


#endif
