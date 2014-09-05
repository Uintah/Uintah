#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ImplicitParticleCreator.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <algorithm>

using namespace Uintah;
using std::vector;
using std::find;

ImplicitParticleCreator::ImplicitParticleCreator(MPMMaterial* matl,
						 MPMLabel* lb,int n8or27) 
  :  ParticleCreator(matl,lb,n8or27)
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

  ParticleVariable<Vector> pacceleration;
  ParticleVariable<double> pvolumeold;
  new_dw->allocateAndPut(pvolumeold, lb->pVolumeOldLabel, subset);
  new_dw->allocateAndPut(pacceleration, lb->pAccelerationLabel, subset);
  
  particleIndex start = 0;

  vector<GeometryObject*>::const_iterator obj;
  for (obj = d_geom_objs.begin(); obj != d_geom_objs.end(); ++obj) {
    particleIndex count = 0;
    GeometryPiece* piece = (*obj)->getPiece();
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getBox();
    Box b = b1.intersect(b2);
    if(b.degenerate()) {
      count = 0;
      continue;
    }

    IntVector ppc = (*obj)->getNumParticlesPerCell();
    Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
    Vector dcorner = dxpp*0.5;
    
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      for(int ix=0;ix < ppc.x(); ix++){
	for(int iy=0;iy < ppc.y(); iy++){
	  for(int iz=0;iz < ppc.z(); iz++){
	    IntVector idx(ix, iy, iz);
	    Point p = lower + dxpp*idx;
	    IntVector cell_idx = iter.index();
	    if(piece->inside(p)){
	      pacceleration[start+count]=Vector(0.,0.,0.);
	      pvolumeold[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
	      count++;
	    }  // if inside
	  }  // loop in z
	}  // loop in y
      }  // loop in x
    } // for
    start += count;
  }


  return subset;
}

particleIndex 
ImplicitParticleCreator::countParticles(const Patch* patch,
					vector<GeometryObject*>& d_geom_objs) const
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
ImplicitParticleCreator::countParticles(GeometryObject* obj,
					const Patch* patch) const
{
  return ParticleCreator::countParticles(obj,patch);
}

void
ImplicitParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)
{
  particle_state.push_back(lb->pVolumeOldLabel);
  particle_state_preReloc.push_back(lb->pVolumeOldLabel_preReloc);

  particle_state.push_back(lb->pAccelerationLabel);
  particle_state_preReloc.push_back(lb->pAccelerationLabel_preReloc);
    
  // This is a hack to remove the pTemperatureLabel from particle_state.
  // Temperature is initialized in the ParticleCreator class, but for
  // the implicit algorithm, we don't use it, hence, we can just remove
  // it from the permanentParticleState, and relocParticles will ignore it.
  
  vector<const VarLabel* >::iterator result;
  
  result = find(particle_state.begin(), particle_state.end(),
		lb->pTemperatureLabel);
  
  particle_state.erase(result);

  result = find(particle_state_preReloc.begin(),particle_state_preReloc.end(),
		lb->pTemperatureLabel_preReloc);
  
  particle_state_preReloc.erase(result);
}
