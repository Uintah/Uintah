
/*
 *  FieldReader.h: Read a field from a file...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_FieldReader_h
#define SCI_project_module_FieldReader_h

#include <UserModule.h>
class Field3DOPort;

class FieldReader : public UserModule {
    Field3DOPort* outfield;
    clString filename;
public:
    FieldReader();
    FieldReader(const FieldReader&, int deep);
    virtual ~FieldReader();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
