
/*
 *  Task.h: Interface to the multitasking library.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Multitask_Task_h
#define SCI_Multitask_Task_h 1

#include <SCICore/share/share.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace SCICore {
namespace Multitask {

const int Task_max_specific = 10;

class Task;
typedef int TaskKey;
class TaskPrivate;

struct SCICORESHARE TaskTime {
    long secs;
    long usecs;
    TaskTime(int secs, int usecs);
    TaskTime(float secs);
    TaskTime(double secs);
    TaskTime();
};

struct SCICORESHARE TaskInfo
{
    int ntasks;
    struct Info {
	const char* name;
	int pid;
	int stacksize;
	int stackused;
	Task* taskid;
    };
    Info* tinfo;
    TaskInfo(int);
    ~TaskInfo();
};

// Basic Task class.  Inherit to provide the body()
class SCICORESHARE Task {
protected:
    friend class TaskManager;
    const char* name;
    int activated;
    int priority;
    int detached;
    void* specific[Task_max_specific];

    int startup(int);
#ifdef _WIN32
	friend unsigned __stdcall runbody(void*);
#else
    friend void* runbody(void*);
#endif

    // Used to implement timers
    struct ITimer {
	TaskTime start;
	TaskTime interval;
	void (*handler)(void*);
	void* cbdata;
	int id;
    };
    ITimer** timers;
    int timer_id;
    int ntimers;
public:
    TaskPrivate* priv;
public:
    enum {DEFAULT_PRIORITY=100};

    // Creation and destruction of tasks
    Task(const char* name, int detached=1, int priority=DEFAULT_PRIORITY);
    virtual ~Task();

    // Overload this to make the task do something
    virtual int body(int)=0;
    
    // This must be called to actually start the task
    void activate(int arg);

    // Used to prematurely interrupt the task
    static void taskexit(Task*, int);

    // To get the Task pointer
    static Task* self();
    const char* get_name();

    // Priority control
    int set_priority(int);
    int get_priority(); // Returns old priority;

    // Give up control to another thread
    static void yield();

    // Wait for other tasks to finish...
    static int wait_for_task(Task*);

    // Timer stuff
    static void sleep(const TaskTime&); // Sleep for this amount of time
    int start_itimer(const TaskTime& start, const TaskTime& interval,
			    void (*handler)(void*), void* data);
    void cancel_itimer(int);

    // O/S Interface
    static void coredump(Task*);
    static void debug(Task*);

    // System interface for thread concurrency control
    static void set_concurrency();
    static void get_concurrency();
    static int nprocessors();
    static void multiprocess(int ntasks, void (*startfn)(void*, int), void* data, bool block=true);

    static void exit_all(int code);
    
    // Used only by main()
    static void initialize(char* progname);
    static void main_exit();

    // Statistics
    static TaskInfo* get_taskinfo();
};

} // End namespace Multitask
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:38  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:07  mcq
// Initial commit
//
// Revision 1.5  1999/07/07 21:11:01  dav
// added beginnings of support for g++ compilation
//
// Revision 1.4  1999/06/30 21:49:04  moulding
// added SHARE to enable win32 shared libraries (dll's).
// .
//
// Revision 1.3  1999/05/06 19:56:21  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:29  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
