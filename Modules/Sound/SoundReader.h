
/*
 *  SoundReader.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SoundReader_h
#define SCI_project_module_SoundReader_h

#include <Module.h>
class SoundOPort;

class SoundReader : public Module {
    SoundOPort* osound;
    clString filename;
public:
    SoundReader();
    SoundReader(const SoundReader&, int deep);
    virtual ~SoundReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
