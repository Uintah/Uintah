
/*
 *  HelpUI.cc: Abstract interface to mosaic...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/String.h>
#include <Dataflow/HelpUI.h>
#include <Multitask/Task.h>

#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

#define MOSAIC_COMMAND "Mosaic"
static int mosaic_running=0;
static int mosaic_pid=0;

class MosaicTask : public Task {
    const clString initial;
public:
    MosaicTask(const clString& initial);
    virtual ~MosaicTask();

    virtual int body(int);
};

HelpUI::HelpUI()
{
}

void HelpUI::load(const clString& name)
{
    char* val=getenv("PROJECT_HOME");
    if(!val){
	val="..";
    }
    char buf[1000];
    sprintf(buf, "%s/help/%s.html", val, name());

    // Start up mosaic if it's not already running...
    if(!mosaic_running){
	MosaicTask* mtask=new MosaicTask(clString(buf));
	mtask->activate(0);
    } else {
	// Send it a signal...
	{
	    char mname[1000];
	    sprintf(mname, "/tmp/Mosaic.%d", mosaic_pid);
	    ofstream command(mname);
	    command << "goto\n";
	    char wd[1000];
	    getcwd(wd, 1000);
	    if(buf[0]=='/'){
		command << "file://localhost" << buf << endl;
	    } else {
		command << "file://localhost/" << wd << "/" << buf << endl;
	    }
	}
	kill(mosaic_pid, SIGUSR1);
    }
}

MosaicTask::MosaicTask(const clString& initial)
: Task("Mosaic - watcher", 1), initial(initial)
{
}

MosaicTask::~MosaicTask()
{
}

int MosaicTask::body(int)
{
    mosaic_running=1;
    pid_t pid=fork();
    int status;
    switch(pid){
    case -1:
	perror("fork");
	exit(-1);
	break;
    case 0:
	// Child...
	execlp(MOSAIC_COMMAND, "-home", initial(), 0);
	cerr << "Error exeuting mosaic!\n";
	perror("execl");
	exit(-1);
	break;
    default:
	// Parent...
	mosaic_pid=pid;
	wait(&status); // We only have one child...
	break;
    }
    cerr << "Mosaic exited with return status: " << status << endl;
    mosaic_running=0;
    return 0;
}
