#include "TensorParticles.h"
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
    return scinew TensorParticles;
}

PersistentTypeID TensorParticles::type_id("TensorParticles", "ParticleSet", maker);
#define TensorParticles_VERSION 3
void TensorParticles::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("TensorParticles::io(Piostream&)");
}

TensorParticles::TensorParticles()
{
}

TensorParticles::TensorParticles(const ParticleVariable<Point>& positions,
			       const ParticleVariable<Matrix3>& tensors,
			       void* callbackClass) :
  positions(positions), tensors(tensors), cbClass(callbackClass)
{
}


TensorParticles::~TensorParticles()
{
}


} // end namespace Datatypes
} // end namespace Kurt
