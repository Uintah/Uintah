#include "FractureParticleCreator.h"

using namespace Uintah;

FractureParticleCreator::FractureParticleCreator()
{
}

FractureParticleCreator::~FractureParticleCreator()
{
}

void FractureParticleCreator::createParticles(MPMMaterial* matl,
					      particleIndex numParticles,
					      CCVariable<short int>& cellNAPID,
					      const Patch*,
					      DataWarehouse* new_dw,
					      MPMLabel* lb,
					      std::vector<GeometryObject*>& d_geom_objs)
{

}

particleIndex FractureParticleCreator::countParticles(const Patch* patch,
						      std::vector<GeometryObject*>& d_geom_objs) const
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex FractureParticleCreator::countParticles(GeometryObject* obj,
						      const Patch* patch) const
{

  return ParticleCreator::countParticles(obj,patch);
}
