
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

const int Task_max_specific = 10;

class Task;
typedef int TaskKey;
class TaskPrivate;

#include <Classlib/String.h>
#include <unistd.h>

struct TaskTime {
    long secs;
    long usecs;
    TaskTime(int secs, int usecs);
    TaskTime(float secs);
    TaskTime(double secs);
    TaskTime();
};

struct TaskInfo
{
    int ntasks;
    struct Info {
	clString name;
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
class Task {
protected:
    friend class TaskManager;
    clString name;
    int activated;
    int priority;
    int detached;
    void* specific[Task_max_specific];

    int startup(int);
    friend void* runbody(void*);

    // Used to implement timers
    struct ITimer {
	TaskTime start;
	TaskTime interval;
	void (*handler)(void*);
	void* cbdata;
	int id;
    };
    ITimer** timers;
    int ntimers;
    int timer_id;
public:
    TaskPrivate* priv;
public:
    enum {DEFAULT_PRIORITY=100};

    // Creation and destruction of tasks
    Task(const clString& name, int detached=1, int priority=DEFAULT_PRIORITY);
    virtual ~Task();

    // Overload this to make the task do something
    virtual int body(int)=0;
    
    // This must be called to actually start the task
    void activate(int arg);

    // Used to prematurely interrupt the task
    static void taskexit(Task*, int);

    // To get the Task pointer
    static Task* self();
    clString get_name();

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

    // Interface to select...
    static int mtselect(int, fd_set*, fd_set*, fd_set*, struct timeval*);

    // O/S Interface
    static void coredump(Task*);
    static void debug(Task*);

    // System interface for thread concurrency control
    static void set_concurrency();
    static void get_concurrency();
    static int nprocessors();

    static void exit_all(int code);
    
    // Used only by main()
    static void initialize(char* progname);
    static void main_exit();

    // Statistics
    static TaskInfo* get_taskinfo();
};

#endif
