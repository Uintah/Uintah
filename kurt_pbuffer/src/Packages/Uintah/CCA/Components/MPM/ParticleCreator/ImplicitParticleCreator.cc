#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/FileGeometryPiece.h>
#include <algorithm>

using namespace Uintah;
using std::vector;
using std::find;

ImplicitParticleCreator::ImplicitParticleCreator(MPMMaterial* matl,
                                                 MPMLabel* lb,
                                                 MPMFlags* flags)
  :  ParticleCreator(matl,lb,flags)
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

  vector<const VarLabel*>::iterator r3,r4;

  if(d_useLoadCurves){
    r3 = find(particle_state.begin(), particle_state.end(),
              lb->pLoadCurveIDLabel);
    particle_state.erase(r3);

    r4 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
              lb->pLoadCurveIDLabel_preReloc);
    particle_state_preReloc.erase(r4);
  }

  vector<const VarLabel*>::iterator r5,r6;
  r5 = find(particle_state.begin(), particle_state.end(),lb->pErosionLabel);
  particle_state.erase(r5);

  r6 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pErosionLabel_preReloc);
  particle_state_preReloc.erase(r6);

  vector<const VarLabel*>::iterator r11,r12;
  r11  = find(particle_state.begin(), particle_state.end(),
             lb->pTempPreviousLabel);
  particle_state.erase(r11);

  r12 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pTempPreviousLabel_preReloc);
  particle_state_preReloc.erase(r12);

}
