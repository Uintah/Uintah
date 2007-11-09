#include <CCA/Components/MPM/ParticleCreator/DefaultParticleCreator.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Grid/Box.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/GeometryPiece/GeometryPiece.h>

using namespace Uintah;


DefaultParticleCreator::DefaultParticleCreator(MPMMaterial* matl, 
                                               MPMFlags* flags)
  :  ParticleCreator(matl,flags)
{
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
					vector<GeometryObject*>& d_geom_objs)
{

  ParticleSubset* subset = ParticleCreator::createParticles(matl,numParticles,
							    cellNAPID,patch,
							    new_dw,
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


