#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
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

  ParticleVariable<Vector> pacceleration;
  ParticleVariable<double> pvolumeold;
  new_dw->allocateAndPut(pvolumeold,    lb->pVolumeOldLabel,    subset);
  new_dw->allocateAndPut(pacceleration, lb->pAccelerationLabel, subset);
  
//  Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();

#if 0
  for(ParticleSubset::iterator iter =  subset->begin();
                               iter != subset->end(); iter++){
    particleIndex idx = *iter;

    pacceleration[idx] = Vector(0.,0.,0.);
//    pvolumeold[idx]    = dxpp.x()*dxpp.y()*dxpp.z();
    pvolumeold[idx]    = 1.0;
  }
#endif

#if 1  // Is there a better way to do this?
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
#endif

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

  // Remove the pSp_volLabel, pSp_volLabel_preReloc,pdisplacement
  vector<const VarLabel*>::iterator r1,r2,r3,r4;

  r1 = find(particle_state.begin(), particle_state.end(),lb->pSp_volLabel);
  particle_state.erase(r1);

  r3 = find(particle_state.begin(), particle_state.end(),lb->pDispLabel);
  particle_state.erase(r3);

  r2 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pSp_volLabel_preReloc);
  particle_state_preReloc.erase(r2);
  
  r4 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
	 lb->pDispLabel_preReloc);
  particle_state_preReloc.erase(r4);
  
}
