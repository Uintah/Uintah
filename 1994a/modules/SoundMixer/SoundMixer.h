
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

#include <UserModule.h>
class MUI_slider_real;
class SoundIPort;
class SoundOPort;

struct SoundMixer_PortInfo {
    double gain;
    MUI_slider_real* interface;
    SoundIPort* isound;
};

class SoundMixer : public UserModule {
    double overall_gain;
    SoundOPort* osound;
    Array1<SoundMixer_PortInfo*> portinfo;
public:
    SoundMixer();
    SoundMixer(const SoundMixer&, int deep);
    virtual ~SoundMixer();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

#endif
