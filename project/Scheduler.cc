
/*
 *  Scheduler.cc: The one that controls the world...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Scheduler.h>

Scheduler::Scheduler(Network* net)
: Task("Scheduler", 1), net(net)
{
}

Scheduler::~Scheduler()
{
}

void Scheduler::set_gui(NetworkEditor* s_gui)
{
    gui=s_gui;
}

int Scheduler::body(int)
{
    return 0;
}
