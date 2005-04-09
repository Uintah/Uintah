
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

#include <Multitask/Task.h>
class Module;

class ModuleHelper : public Task {
    Module* module;
public:
    ModuleHelper(Module* module);
    virtual ~ModuleHelper();

    virtual int body(int);
};

#endif
