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

#include <Core/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>

namespace SCIRun {

class SCICORESHARE TCLTask : public Runnable {
    int argc;
    char** argv;
    Semaphore cont;
    Semaphore start;
protected:
    virtual void run();
    friend void wait_func(void*);
    void mainloop_wait();
public:
    TCLTask(int argc, char* argv[]);
    virtual ~TCLTask();
    static Thread* get_owner();
    static void lock();
    static int try_lock();
    static void unlock();
    void mainloop_waitstart();
    void release_mainloop();
};

} // End namespace SCIRun


#endif
