//static char *id="@(#) $Id$";

/*
 *  ParticleSet.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Datatypes/Particles/ParticleSet.h>
#include <Containers/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace Uintah {
namespace Datatypes {

PersistentTypeID ParticleSet::type_id("ParticleSet", "Datatype", 0);

ParticleSet::ParticleSet()
{
}

ParticleSet::ParticleSet(const ParticleSet&)
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

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/07/27 16:59:00  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//


