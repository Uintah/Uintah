
/*
 *  DebugSettings.h: Debug settings visualizer
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DebugSettings_h
#define SCI_project_DebugSettings_h 1

#include <Containers/Array1.h>
#include <TclInterface/TCL.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::Array1;

class TCLvarintp;

class DebugSettings : public TCL {
    Array1<TCLvarintp*> variables;
public:
    DebugSettings();
    ~DebugSettings();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:13  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:22  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:31  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
