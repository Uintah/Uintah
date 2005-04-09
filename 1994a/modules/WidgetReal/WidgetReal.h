
/*
 *  WidgetReal.h:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_WidgetReal_h
#define SCI_project_module_WidgetReal_h

#include <UserModule.h>

class WidgetReal : public UserModule {
public:
    WidgetReal();
    WidgetReal(const WidgetReal&, int deep);
    virtual ~WidgetReal();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
