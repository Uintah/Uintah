//static char *id="@(#) $Id$";

/*
 *  ParticleGridReader.cc
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Datatypes/Particles/ParticleGridReader.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Malloc/Allocator.h>

#include <iostream.h>

namespace Uintah {
namespace Datatypes {

PersistentTypeID ParticleGridReader::type_id("ParticleGridReader", "Datatype", 0);

ParticleGridReader::ParticleGridReader()
{
}

ParticleGridReader::ParticleGridReader(const ParticleGridReader&)
{
}

ParticleGridReader::~ParticleGridReader()
{
}

#define PARTICLEGRIDREADER_VERSION 1

void ParticleGridReader::io(Piostream& stream)
{
  stream.begin_class("ParticleGridReader", PARTICLEGRIDREADER_VERSION);
    stream.end_class();
}

void ParticleGridReader::SetFile( const clString& /*file*/ )
{
    NOT_FINISHED("ParticlGridReader::SetFile");
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/07/27 16:58:59  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 23:18:39  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
