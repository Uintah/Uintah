#include <CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Box.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <algorithm>

using namespace Uintah;
using std::vector;
using std::find;

#define HEAT
//#undef HEAT

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
#ifdef HEAT
  pExternalHeatFlux[i] = 0.;
#endif
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
#ifdef HEAT
  new_dw->allocateAndPut(pExternalHeatFlux, d_lb->pExternalHeatFluxLabel, 
                         subset);
#endif

  return subset;

}

void
ImplicitParticleCreator::registerPermanentParticleState(MPMMaterial* /*matl*/)

{
  particle_state.push_back(d_lb->pAccelerationLabel);
  particle_state_preReloc.push_back(d_lb->pAccelerationLabel_preReloc);

#ifdef HEAT
  particle_state.push_back(d_lb->pExternalHeatFluxLabel);
  particle_state_preReloc.push_back(d_lb->pExternalHeatFluxLabel_preReloc);
#endif

#if 0
  vector<const VarLabel*>::iterator r3,r4;

  if(d_useLoadCurves){
    r3 = find(particle_state.begin(), particle_state.end(),
              d_lb->pLoadCurveIDLabel);
    particle_state.erase(r3);

    r4 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
              d_lb->pLoadCurveIDLabel_preReloc);
    particle_state_preReloc.erase(r4);
  }
#endif
}
