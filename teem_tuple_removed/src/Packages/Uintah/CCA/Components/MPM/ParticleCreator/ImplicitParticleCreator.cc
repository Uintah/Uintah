#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <algorithm>

using namespace Uintah;
using std::vector;
using std::find;

ImplicitParticleCreator::ImplicitParticleCreator(MPMMaterial* matl,
                                                 MPMLabel* lb,
                                                 int n8or27,
                                                 bool haveLoadCurve,
						 bool doErosion) 
  :  ParticleCreator(matl,lb,n8or27,haveLoadCurve, doErosion)
{
  registerPermanentParticleState(matl,lb);

  // Transfer to the lb's permanent particle state array of vectors

  lb->d_particleState.push_back(particle_state);
  lb->d_particleState_preReloc.push_back(particle_state_preReloc);
}

ImplicitParticleCreator::~ImplicitParticleCreator()
{
}

ParticleSubset* 
ImplicitParticleCreator::createParticles(MPMMaterial* matl, 
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

void 
ImplicitParticleCreator::initializeParticle(const Patch* patch,
					    vector<GeometryObject*>::const_iterator obj,
					    MPMMaterial* matl,
					    Point p, IntVector cell_idx,
					    particleIndex i,
					    CCVariable<short int>& cellNAPI)
{

  ParticleCreator::initializeParticle(patch,obj,matl,p,cell_idx,i,cellNAPI);

  pacceleration[i] = Vector(0.,0.,0.);
  pvolumeold[i] = pvolume[i];
}

particleIndex 
ImplicitParticleCreator::countParticles(const Patch* patch,
					vector<GeometryObject*>& d_geom_objs) 
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
ImplicitParticleCreator::countAndCreateParticles(const Patch* patch,
						 GeometryObject* obj) 
{
  return ParticleCreator::countAndCreateParticles(patch,obj);
}


ParticleSubset* 
ImplicitParticleCreator::allocateVariables(particleIndex numParticles, 
					   int dwi,MPMLabel* lb, 
					   const Patch* patch,
					   DataWarehouse* new_dw)
{

  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
							      dwi,lb,patch,
							      new_dw);

  new_dw->allocateAndPut(pvolumeold,    lb->pVolumeOldLabel,    subset);
  new_dw->allocateAndPut(pacceleration, lb->pAccelerationLabel, subset);

  return subset;

}

void
ImplicitParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)
{
  particle_state.push_back(lb->pVolumeOldLabel);
  particle_state_preReloc.push_back(lb->pVolumeOldLabel_preReloc);

  particle_state.push_back(lb->pAccelerationLabel);
  particle_state_preReloc.push_back(lb->pAccelerationLabel_preReloc);

  // Remove the pSp_volLabel, pSp_volLabel_preReloc,pdisplacement
  vector<const VarLabel*>::iterator r1,r2;

  r1 = find(particle_state.begin(), particle_state.end(),lb->pSp_volLabel);
  particle_state.erase(r1);

  r2 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pSp_volLabel_preReloc);
  particle_state_preReloc.erase(r2);
}
