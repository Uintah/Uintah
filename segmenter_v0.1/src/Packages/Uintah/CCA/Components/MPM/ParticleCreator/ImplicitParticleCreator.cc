#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
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
                                                 MPMFlags* flags)
  :  ParticleCreator(matl,flags)
{
  registerPermanentParticleState(matl);
}

ImplicitParticleCreator::~ImplicitParticleCreator()
{
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
}


ParticleSubset* 
ImplicitParticleCreator::allocateVariables(particleIndex numParticles, 
                                           int dwi,const Patch* patch,
                                           DataWarehouse* new_dw)
{

  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
                                                              dwi,patch,
                                                              new_dw);

  new_dw->allocateAndPut(pacceleration, d_lb->pAccelerationLabel, subset);

  return subset;

}

void
ImplicitParticleCreator::registerPermanentParticleState(MPMMaterial* /*matl*/)

{
  particle_state.push_back(d_lb->pAccelerationLabel);
  particle_state_preReloc.push_back(d_lb->pAccelerationLabel_preReloc);

  vector<const VarLabel*>::iterator r3,r4;

  if(d_useLoadCurves){
    r3 = find(particle_state.begin(), particle_state.end(),
              d_lb->pLoadCurveIDLabel);
    particle_state.erase(r3);

    r4 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
              d_lb->pLoadCurveIDLabel_preReloc);
    particle_state_preReloc.erase(r4);
  }
}
