/*----------------------------------------------------------------------
CLASS
    ParticleSetExtensionPort

    
OVERVIEW TEXT
    


KEYWORDS

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 8, 1999
----------------------------------------------------------------------*/


#ifndef SCI_project_ParticleSetExtensionPort_h
#define SCI_project_ParticleSetExtensionPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ParticleSetExtension.h>

typedef SimpleIPort<ParticleSetExtensionHandle> ParticleSetExtensionIPort;
typedef SimpleOPort<ParticleSetExtensionHandle> ParticleSetExtensionOPort;

#endif
