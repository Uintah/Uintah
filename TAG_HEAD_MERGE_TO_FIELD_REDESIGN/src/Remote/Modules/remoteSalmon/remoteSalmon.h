
/*
 *  Salmon.h: The Geometry Viewer!
 *
 *  Written by:
 *   Steven G. Parker & Dave Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_remoteSalmon_h
#define SCI_project_module_remoteSalmon_h

#include <queue>
#include <iostream>
#include <SCICore/Containers/String.h>
#include <PSECommon/Modules/Salmon/Salmon.h>

namespace Remote {
namespace Modules {

using PSECommon::Modules::Salmon;
using SCICore::Containers::clString;

using namespace SCICore::TclInterface;

using namespace std;

//----------------------------------------------------------------------
class remoteSalmon : public Salmon {
    
public:

    remoteSalmon(const clString& id);

    void tcl_command(TCLArgs&, void*);

};

} // End namespace Modules
} // End namespace Remote

#endif // SCI_project_module_remoteSalmon_h
