
/*
 *  ModuleHelper.h:  Thread to execute modules..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ModuleHelper_h
#define SCI_project_ModuleHelper_h 1

#include <SCICore/Thread/Runnable.h>
#include <PSECore/share/share.h>

namespace PSECore {
namespace Dataflow {

class Module;

class PSECORESHARE ModuleHelper : public SCICore::Thread::Runnable {
    Module* module;
public:
    ModuleHelper(Module* module);
    virtual ~ModuleHelper();

    virtual void run();
};

} // End namespace Dataflow
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/08/28 17:54:29  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/26 23:59:07  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:23  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:58  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 22:02:43  dav
// added back .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
