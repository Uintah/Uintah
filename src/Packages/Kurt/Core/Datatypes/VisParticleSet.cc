#include "VisParticleSet.h"
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

namespace Kurt {
using Uintah::DataArchive;
using Uintah::ParticleVariable;

using namespace SCIRun;


static Persistent* maker()
{
    return scinew VisParticleSet;
}

PersistentTypeID VisParticleSet::type_id("VisParticleSet", "ParticleSet", maker);
#define VisParticleSet_VERSION 3
void VisParticleSet::io(Piostream&)
{
    NOT_FINISHED("VisParticleSet::io(Piostream&)");
}

VisParticleSet::VisParticleSet()
{
}

VisParticleSet::VisParticleSet(const ParticleVariable<Point>& positions,
			       const ParticleVariable<double>& scalars,
			       const ParticleVariable<Vector>& vectors,
			       void* callbackClass) :
  positions(positions), scalars(scalars), vectors(vectors),
  cbClass(callbackClass)
{
}


VisParticleSet::~VisParticleSet()
{
}

} // End namespace Kurt

