#include "DefaultParticleCreator.h"
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

using namespace Uintah;


DefaultParticleCreator::DefaultParticleCreator()
{
}

DefaultParticleCreator::~DefaultParticleCreator()
{
}

void DefaultParticleCreator::createParticles(MPMMaterial* matl,
					     particleIndex numParticles,
					     CCVariable<short int>& cellNAPID,
					     const Patch* patch,
					     DataWarehouse* new_dw,
					     MPMLabel* lb,
					     std::vector<GeometryObject*>& d_geom_objs)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,
							patch);
  ParticleVariable<Point> position;
  ParticleVariable<Vector> pvelocity,pexternalforce,psize;
  ParticleVariable<double> pmass,pvolume,ptemperature;
  ParticleVariable<long64> pparticleID;
  new_dw->allocateAndPut(position, lb->pXLabel, subset);
  new_dw->allocateAndPut(pvelocity, lb->pVelocityLabel, subset); 
  new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass, lb->pMassLabel, subset);
  new_dw->allocateAndPut(pvolume, lb->pVolumeLabel, subset);
  new_dw->allocateAndPut(ptemperature, lb->pTemperatureLabel, subset);
  new_dw->allocateAndPut(pparticleID, lb->pParticleIDLabel, subset);
  new_dw->allocateAndPut(psize, lb->pSizeLabel, subset);

  particleIndex start = 0;
  
  for (int i=0; i<(int)d_geom_objs.size();i++) {
    particleIndex count = 0;
    GeometryObject* obj = d_geom_objs[i];
    GeometryPiece* piece = obj->getPiece();
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getBox();
    Box b = b1.intersect(b2);
    if(b.degenerate()) {
      count = 0;
      continue;
    }
    
    IntVector ppc = obj->getNumParticlesPerCell();
    Vector dxpp = patch->dCell()/obj->getNumParticlesPerCell();
    Vector dcorner = dxpp*0.5;
    // Size as a fraction of the cell size
    Vector size(1./((double) ppc.x()),
		1./((double) ppc.y()),
		1./((double) ppc.z()));
    
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      for(int ix=0;ix < ppc.x(); ix++){
	for(int iy=0;iy < ppc.y(); iy++){
	  for(int iz=0;iz < ppc.z(); iz++){
	    IntVector idx(ix, iy, iz);
	    Point p = lower + dxpp*idx;
	    IntVector cell_idx = iter.index();
	    // If the assertion fails then we may just need to change
	    // the format of particle ids such that the cell indices
	    // have more bits.
	    ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
		   && cell_idx.z() <= 0xffff);
	    long64 cellID = ((long64)cell_idx.x() << 16) |
	      ((long64)cell_idx.y() << 32) |
	      ((long64)cell_idx.z() << 48);
	    if(piece->inside(p)){
	      position[start+count]=p;
	      pvolume[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
	      pvelocity[start+count]=obj->getInitialVelocity();
	      ptemperature[start+count]=obj->getInitialTemperature();
	      pmass[start+count]=
		matl->getInitialDensity() * pvolume[start+count];
	      // Determine if particle is on the surface
	      pexternalforce[start+count]=Vector(0,0,0); // for now
	      short int& myCellNAPID = cellNAPID[cell_idx];
	      pparticleID[start+count] = cellID | (long64)myCellNAPID;
	      psize[start+count] = size;
	      ASSERT(myCellNAPID < 0x7fff);
	      myCellNAPID++;
	      count++;
	    }  // if inside
	  }  // loop in z
	}  // loop in y
      }  // loop in x
    } // for
    start += count;
  }

  ParticleCreator::applyForceBC(start,pexternalforce,pmass,position);

}

particleIndex DefaultParticleCreator::countParticles(const Patch* patch,
						     std::vector<GeometryObject*>& d_geom_objs) const
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex DefaultParticleCreator::countParticles(GeometryObject* obj,
						      const Patch* patch) const
{

  return ParticleCreator::countParticles(obj,patch);
}


