
/*
 *  GenSurface.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_GenSurface_h
#define SCI_project_module_GenSurface_h

#include <UserModule.h>

class GenSurface : public UserModule {
public:
    GenSurface();
    GenSurface(const GenSurface&, int deep);
    virtual ~GenSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
