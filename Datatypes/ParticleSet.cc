
/*
 *  ParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Datatypes/ParticleSet.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

PersistentTypeID ParticleSet::type_id("ParticleSet", "Datatype", 0);

ParticleSet::ParticleSet()
{
}

ParticleSet::ParticleSet(const ParticleSet& c)
{
}

ParticleSet::~ParticleSet()
{
}

#define PARTICLESET_VERSION 1

void ParticleSet::io(Piostream& stream)
{
    stream.begin_class("ParticleSet", PARTICLESET_VERSION);
    stream.end_class();
}
