#include "FractureParticleCreator.h"

using namespace Uintah;
using std::vector;

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
					      vector<GeometryObject*>& d_geom_objs)
{

  //ParticleCreator::applyForceBC(start,pexternalforce,pmass,position);

}

particleIndex FractureParticleCreator::countParticles(const Patch* patch,
						      vector<GeometryObject*>& d_geom_objs) const
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex FractureParticleCreator::countParticles(GeometryObject* obj,
						      const Patch* patch) const
{

  return ParticleCreator::countParticles(obj,patch);
}
