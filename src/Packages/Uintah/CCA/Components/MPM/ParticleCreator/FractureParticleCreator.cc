#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

using namespace Uintah;
using std::vector;

FractureParticleCreator::FractureParticleCreator(MPMMaterial* matl,
						 MPMLabel* lb,
                                                 int n8or27,
                                                 bool haveLoadCurve) 
  :  ParticleCreator(matl,lb,n8or27,haveLoadCurve)
{
  registerPermanentParticleState(matl,lb);

  // Transfer to the lb's permanent particle state array of vectors

  lb->d_particleState.push_back(particle_state);
  lb->d_particleState_preReloc.push_back(particle_state_preReloc);
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

 ParticleVariable<Point> position0;
 constParticleVariable<Point> position;

 new_dw->allocateAndPut(position0,lb->pX0Label,subset);
 new_dw->get(position,lb->pXLabel,subset);

 position0.copyData(position);

 return subset;
}

particleIndex 
FractureParticleCreator::countParticles(const Patch* patch,
					vector<GeometryObject*>& d_geom_objs) const
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex FractureParticleCreator::countParticles(GeometryObject* obj,
						      const Patch* patch) const
{

  return ParticleCreator::countParticles(obj,patch);
}


void
FractureParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)
{
  particle_state.push_back(lb->pX0Label);
  particle_state_preReloc.push_back(lb->pX0Label_preReloc);

}
