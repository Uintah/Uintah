#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

using namespace Uintah;
using std::vector;

FractureParticleCreator::FractureParticleCreator(MPMMaterial* matl,
						 MPMLabel* lb,
						 MPMFlags* flags)
  :  ParticleCreator(matl,lb,flags)
{
  registerPermanentParticleState(matl,lb);
  d_fracture = flags->d_fracture = true;

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

 //ParticleVariable<Point> position0;
 //constParticleVariable<Point> position;

 //new_dw->allocateAndPut(position0,lb->pX0Label,subset);
 //new_dw->get(position,lb->pXLabel,subset);

 //position0.copyData(position);

 return subset;
}

particleIndex 
FractureParticleCreator::countParticles(const Patch* patch,
					vector<GeometryObject*>& d_geom_objs) 
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
FractureParticleCreator::countAndCreateParticles(const Patch* patch,
						 GeometryObject* obj) 
{

  return ParticleCreator::countAndCreateParticles(patch,obj);
}


void
FractureParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)
{
  //particle_state.push_back(lb->pX0Label);
  //particle_state_preReloc.push_back(lb->pX0Label_preReloc);

  vector<const VarLabel*>::iterator r1,r2;
  r1 = find(particle_state.begin(), particle_state.end(),lb->pErosionLabel);
  particle_state.erase(r1);

  r2 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pErosionLabel_preReloc);
  particle_state_preReloc.erase(r2);
}
