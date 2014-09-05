/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
