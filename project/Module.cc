
/*
 *  Module.cc: Basic implementation of modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Module.h>
#include <NotFinished.h>

Module::Module(const clString& name)
: name(name), xpos(10), ypos(10),
  interface_initialized(0), state(NeedData),
  mailbox(10)
{
}

Module::Module(const Module& copy, int deep)
: name(copy.name), xpos(copy.xpos+10), ypos(copy.ypos+10),
  interface_initialized(0),
  state(NeedData),
  mailbox(10)
{
}

Module::~Module()
{
}

clString Module::get_name()
{
    return name;
}

int Module::body(int)
{
    state=Executing;
    timer.clear();
    timer.start();
    execute();
    timer.stop();
    state=Completed;
    NOT_FINISHED("Module::body");
    return 0;
}

void Module::update_progress(double p)
{
    progress=p;
    need_update=1;
}

void Module::update_progress(int n, int max)
{
    progress=double(n)/double(max);
    need_update=1;
}

double Module::get_execute_time()
{
    return timer.time();
}

int Module::needs_update()
{
    return need_update;
}

void Module::updated()
{
    need_update=0;
}

double Module::get_progress()
{
    return progress;
}

Module::State Module::get_state()
{
    return state;
}

int Module::niports()
{
    return iports.size();
}

IPort* Module::iport(int i)
{
    return iports[i];
}

int Module::noports()
{
    return oports.size();
}

OPort* Module::oport(int i)
{
    return oports[i];
}

void Module::activate()
{
}

ModuleMsg::ModuleMsg(Module::ConnectionMode mode, int output, int port,
		     Module* tomod, int toport, Connection* connection)
: type(Connect), mode(mode), output(output), port(port),
  tomod(tomod), toport(toport), connection(connection)
{
}
