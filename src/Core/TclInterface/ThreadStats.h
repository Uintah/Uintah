
/*
 *  ThreadStats.h: Thread information visualizer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ThreadStats_h
#define SCI_project_ThreadStats_h 1

#include <SCICore/TclInterface/TCL.h>

namespace SCICore {
  namespace Multitask {
    struct TaskInfo;
  }
}

namespace SCICore {
namespace TclInterface {

using SCICore::Multitask::TaskInfo;

class SCICORESHARE ThreadStats : public TCL {
    TaskInfo* info;
    TaskInfo* oldinfo;
    int maxstacksize;
public:
    ThreadStats();
    ~ThreadStats();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:17  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:25  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:35  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
