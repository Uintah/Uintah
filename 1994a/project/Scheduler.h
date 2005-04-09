
/*
 *  Scheduler.h: Interface to Scheduler class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Scheduler_h
#define SCI_project_Scheduler_h 1

#include <Multitask/Task.h>
class Network;
class NetworkEditor;

class Scheduler : public Task {
    Network* net;
    NetworkEditor* gui;
public:
    Scheduler(Network*);
    ~Scheduler();
    void set_gui(NetworkEditor*);

    int body(int);

    // These functions just send messages to the scheduler...
    void shutdown_scheduler();
};

#endif /* SCI_project_Scheduler_h */
