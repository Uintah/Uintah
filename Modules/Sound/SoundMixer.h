
/*
 *  SoundMixer.h: The 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundMixer_h
#define SCI_project_module_SoundMixer_h

#include <Module.h>
class SoundIPort;
class SoundOPort;

struct SoundMixer_PortInfo {
    double gain;
    SoundIPort* isound;
};

class SoundMixer : public Module {
    double overall_gain;
    SoundOPort* osound;
    Array1<SoundMixer_PortInfo*> portinfo;
public:
    SoundMixer(const clString& id);
    SoundMixer(const SoundMixer&, int deep);
    virtual ~SoundMixer();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

#endif
