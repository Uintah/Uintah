#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>

using namespace Uintah;
using std::vector;

FractureParticleCreator::FractureParticleCreator(MPMMaterial* matl,
						 MPMLabel* lb,
                                                 int n8or27,
                                                 bool haveLoadCurve) 
  :  ParticleCreator(matl,lb,n8or27,haveLoadCurve)
{
}

FractureParticleCreator::~FractureParticleCreator()
{
}

ParticleSubset* FractureParticleCreator::createParticles(MPMMaterial* matl,
					      particleIndex numParticles,
					      CCVariable<short int>& cellNAPID,
					      const Patch* patch,
					      DataWarehouse* new_dw,
					      MPMLabel* lb,
					      vector<GeometryObject*>& d_geom_objs)
{

 ParticleSubset* subset = ParticleCreator::createParticles(matl,numParticles,
							    cellNAPID,patch,
							    new_dw,lb,
							    d_geom_objs);

  return subset;
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
