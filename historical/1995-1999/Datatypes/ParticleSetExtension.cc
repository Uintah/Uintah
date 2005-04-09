
#include <Datatypes/ParticleSetExtension.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <Classlib/NotFinished.h>
PersistentTypeID ParticleSetExtension::type_id("ParticleSetExtension", "Datatype", 0);

ParticleSetExtension::ParticleSetExtension()
{
}
ParticleSetExtension::ParticleSetExtension( clString scalarVar,
				      clString vectorVar,
				      void* cbClass) :
  sVar( scalarVar ), vVar( vectorVar), cbClass( cbClass )
{
  // No body
}
  

ParticleSetExtension::ParticleSetExtension(const ParticleSetExtension&) :
   sVar(""), vVar(""), cbClass(0)
{
  // No body
}

ParticleSetExtension* ParticleSetExtension::clone() const
{
  return scinew ParticleSetExtension();
}



ParticleSetExtension::~ParticleSetExtension()
{
  // No body
}

void ParticleSetExtension::SetScalarId(const  clString& id)
{
  sVar = id;
}

void ParticleSetExtension::SetVectorId(const  clString& id)
{
  vVar = id;
}

void ParticleSetExtension::SetCallback( void* f )
{
  cbClass = f;
}

const clString& ParticleSetExtension::getScalarId()
{
  return sVar;
}
const clString& ParticleSetExtension::getVectorId()
{
  return vVar;
  }

void* ParticleSetExtension::getCallbackClass()
{
  return cbClass;
}

#define PARTICLESETEXTENSION_VERSION 1

void ParticleSetExtension::io(Piostream& stream)
{
  NOT_FINISHED( "ParticleSetExtension::io" );
  stream.begin_class("ParticleSetExtension", PARTICLESETEXTENSION_VERSION);
    stream.end_class();
}
