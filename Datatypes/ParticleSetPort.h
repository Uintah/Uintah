
/*
 *  ParticleSetPort.h
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_project_ParticleSetPort_h
#define SCI_project_ParticleSetPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ParticleSet.h>

typedef SimpleIPort<ParticleSetHandle> ParticleSetIPort;
typedef SimpleOPort<ParticleSetHandle> ParticleSetOPort;

#endif
