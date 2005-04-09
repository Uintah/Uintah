
/*
 *  CrossFader.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_CrossFader_h
#define SCI_project_module_CrossFader_h

#include <UserModule.h>
class SoundIPort;
class SoundOPort;

class CrossFader : public UserModule {
    SoundIPort* isound1;
    SoundIPort* isound2;
    double fade;
    double gain;
    SoundOPort* osound;
public:
    CrossFader();
    CrossFader(const CrossFader&, int deep);
    virtual ~CrossFader();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
