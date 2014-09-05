#include "VisParticleSet.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

namespace Kurt {
namespace Datatypes {

using Uintah::DataArchive;
using Uintah::ParticleVariable;

using SCICore::Datatypes::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;


static Persistent* maker()
{
    return scinew VisParticleSet;
}

PersistentTypeID VisParticleSet::type_id("VisParticleSet", "ParticleSet", maker);
#define VisParticleSet_VERSION 3
void VisParticleSet::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
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


} // end namespace Datatypes
} // end namespace Kurt
