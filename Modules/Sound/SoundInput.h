
/*
 *  SoundInput.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundInput_h
#define SCI_project_module_SoundInput_h

#include <Module.h>
class SoundOPort;

class SoundInput : public Module {
    SoundOPort* osound;
    int onoff;
    double rate;
public:
    SoundInput();
    SoundInput(const SoundInput&, int deep);
    virtual ~SoundInput();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual int should_execute();
};

#endif
