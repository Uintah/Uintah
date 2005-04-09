
/*
 *  Gradient.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Gradient_h
#define SCI_project_module_Gradient_h

#include <Dataflow/Module.h>

class Gradient : public Module {
public:
    Gradient(const clString& id);
    Gradient(const Gradient&, int deep);
    virtual ~Gradient();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
