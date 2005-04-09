
/*
 *  ApplyBC.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_ApplyBC_h
#define SCI_project_module_ApplyBC_h

#include <UserModule.h>

class ApplyBC : public UserModule {
public:
    ApplyBC();
    ApplyBC(const ApplyBC&, int deep);
    virtual ~ApplyBC();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
