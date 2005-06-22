#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/DefaultParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>

using namespace Uintah;


DefaultParticleCreator::DefaultParticleCreator(MPMMaterial* matl, 
                                               MPMLabel* lb,
                                               MPMFlags* flags,
                                               SimulationStateP& sharedState)
  :  ParticleCreator(matl,lb, flags,sharedState)
{
  // Transfer to the sharedState's permanent particle state array of vectors
  sharedState->d_particleState.push_back(particle_state);
  sharedState->d_particleState_preReloc.push_back(particle_state_preReloc);
}

DefaultParticleCreator::~DefaultParticleCreator()
{
}

ParticleSubset* 
DefaultParticleCreator::createParticles(MPMMaterial* matl,
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

particleIndex 
DefaultParticleCreator::countParticles(const Patch* patch,
				       vector<GeometryObject*>& d_geom_objs) 
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
DefaultParticleCreator::countAndCreateParticles(const Patch* patch,
						GeometryObject* obj) 
{

  return ParticleCreator::countAndCreateParticles(patch,obj);
}


