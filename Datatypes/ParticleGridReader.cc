
#include <Datatypes/ParticleGridReader.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

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
