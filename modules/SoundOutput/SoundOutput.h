
/*
 *  SoundOutput.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundOutput_h
#define SCI_project_module_SoundOutput_h

#include <UserModule.h>
class SoundIPort;

class SoundOutput : public UserModule {
    SoundIPort* isound;
public:
    SoundOutput();
    SoundOutput(const SoundOutput&, int deep);
    virtual ~SoundOutput();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
