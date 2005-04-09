
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

#include <Dataflow/Module.h>
class SoundIPort;

class SoundOutput : public Module {
    SoundIPort* isound;
public:
    SoundOutput(const clString& id);
    SoundOutput(const SoundOutput&, int deep);
    virtual ~SoundOutput();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
