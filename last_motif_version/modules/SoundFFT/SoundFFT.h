
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

#include <UserModule.h>
#include <SoundData.h>
class MUI_slider_real;

class SoundFFT : public UserModule {
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
