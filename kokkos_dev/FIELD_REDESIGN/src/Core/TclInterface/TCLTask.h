 
/*
 *  TCLTask.h:  Handle TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TCLTask_h
#define SCI_project_TCLTask_h 1

#include <SCICore/share/share.h>
#include <SCICore/Thread/Runnable.h>
#include <SCICore/Thread/Semaphore.h>

namespace SCICore {
namespace TclInterface {

class SCICORESHARE TCLTask : public SCICore::Thread::Runnable {
    int argc;
    char** argv;
    SCICore::Thread::Semaphore cont;
    SCICore::Thread::Semaphore start;
protected:
    virtual void run();
    friend void wait_func(void*);
    void mainloop_wait();
public:
    TCLTask(int argc, char* argv[]);
    virtual ~TCLTask();
    static SCICore::Thread::Thread* get_owner();
    static void lock();
    static int try_lock();
    static void unlock();
    void mainloop_waitstart();
    void release_mainloop();
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/28 17:54:52  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:45  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:16  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:24  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:34  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
