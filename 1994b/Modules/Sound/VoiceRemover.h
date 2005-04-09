
/*
 *  VoiceRemover.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_VoiceRemover_h
#define SCI_project_module_VoiceRemover_h

#include <Dataflow/Module.h>
class SoundIPort;
class SoundOPort;

class VoiceRemover : public Module {
    SoundIPort* isound;
    SoundOPort* osound;
public:
    VoiceRemover(const clString& id);
    VoiceRemover(const VoiceRemover&, int deep);
    virtual ~VoiceRemover();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
