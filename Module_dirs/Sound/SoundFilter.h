
/*
 *  SoundFilter.h: The 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundFilter_h
#define SCI_project_module_SoundFilter_h

#include <Module.h>
class SoundIPort;
class SoundOPort;

class SoundFilter : public Module {
    SoundIPort* isound;
    SoundOPort* osound;
    double lower_cutoff;
    double upper_cutoff;
public:
    SoundFilter(const clString& id);
    SoundFilter(const SoundFilter&, int deep);
    virtual ~SoundFilter();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
