
/*
 *  SoundFFT.h: The 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundFFT_h
#define SCI_project_module_SoundFFT_h

#include <Module.h>
#include <SoundData.h>

class SoundFFT : public Module {
    OutSoundData outsound;
    InSoundData isound;

public:
    SoundFFT();
    SoundFFT(const SoundFFT&, int deep);
    virtual ~SoundFFT();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
