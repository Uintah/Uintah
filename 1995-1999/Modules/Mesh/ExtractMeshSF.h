
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

#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/MeshPort.h>

class ExtractMeshSF : public Module {
    ScalarFieldIPort* inport;
    MeshOPort* outport;
public:
    ExtractMeshSF(const clString& id);
    ExtractMeshSF(const ExtractMeshSF&, int deep);
    virtual ~ExtractMeshSF();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
