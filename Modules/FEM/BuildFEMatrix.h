
/*
 *  BuildFEMatrix.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_BuildFEMatrix_h
#define SCI_project_module_BuildFEMatrix_h

#include <Module.h>

class BuildFEMatrix : public Module {
public:
    BuildFEMatrix(const clString& id);
    BuildFEMatrix(const BuildFEMatrix&, int deep);
    virtual ~BuildFEMatrix();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
