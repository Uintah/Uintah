
/*
 *  ExtractMeshSF.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_ExtractMeshSF_h
#define SCI_project_module_ExtractMeshSF_h

#include <UserModule.h>
#include <ScalarFieldPort.h>
#include <MeshPort.h>

class ExtractMeshSF : public UserModule {
    ScalarFieldIPort* inport;
    MeshOPort* outport;
public:
    ExtractMeshSF();
    ExtractMeshSF(const ExtractMeshSF&, int deep);
    virtual ~ExtractMeshSF();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
