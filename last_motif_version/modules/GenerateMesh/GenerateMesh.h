
/*
 *  GenerateMesh.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_GenerateMesh_h
#define SCI_project_module_GenerateMesh_h

#include <UserModule.h>

class GenerateMesh : public UserModule {
public:
    GenerateMesh();
    GenerateMesh(const GenerateMesh&, int deep);
    virtual ~GenerateMesh();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
