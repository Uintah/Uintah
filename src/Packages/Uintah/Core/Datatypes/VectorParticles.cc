#include "VectorParticles.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

namespace Uintah {
namespace Datatypes {

using Uintah::DataArchive;
using Uintah::ParticleVariable;

using SCICore::Datatypes::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;


static Persistent* maker()
{
    return scinew VectorParticles;
}

PersistentTypeID VectorParticles::type_id("VectorParticles", "ParticleSet", maker);
#define VectorParticles_VERSION 3
void VectorParticles::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("VectorParticles::io(Piostream&)");
}

VectorParticles::VectorParticles()
{
}

VectorParticles::VectorParticles(const ParticleVariable<Point>& positions,
			       const ParticleVariable<Vector>& vectors,
			       void* callbackClass) :
  positions(positions), vectors(vectors), cbClass(callbackClass)
{
}


VectorParticles::~VectorParticles()
{
}


} // end namespace Datatypes
} // end namespace Kurt
