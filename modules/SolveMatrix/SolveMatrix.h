
/*
 *  SolveMatrix.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SolveMatrix_h
#define SCI_project_module_SolveMatrix_h

#include <UserModule.h>

class SolveMatrix : public UserModule {
public:
    SolveMatrix();
    SolveMatrix(const SolveMatrix&, int deep);
    virtual ~SolveMatrix();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
