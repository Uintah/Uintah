#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/MembraneParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SphereMembraneGeometryPiece.h>


using namespace Uintah;

MembraneParticleCreator::MembraneParticleCreator(MPMMaterial* matl,
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

MembraneParticleCreator::~MembraneParticleCreator()
{
}

ParticleSubset* MembraneParticleCreator::createParticles(MPMMaterial* matl, 
					      particleIndex numParticles,
					      CCVariable<short int>& cellNAPID,
					      const Patch* patch,
					      DataWarehouse* new_dw,
					      MPMLabel* lb,
					      vector<GeometryObject*>& d_geom_objs)
{
  int dwi = matl->getDWIndex();

  ParticleSubset* subset =  ParticleCreator::allocateVariables(numParticles,
							       dwi,lb,patch,
							       new_dw);
  ParticleVariable<Vector> pTang1,pTang2,pNorm;
  new_dw->allocateAndPut(pTang1, lb->pTang1Label, subset);
  new_dw->allocateAndPut(pTang2, lb->pTang2Label, subset);
  new_dw->allocateAndPut(pNorm, lb->pNormLabel,  subset);
  
  particleIndex start = 0;

  vector<GeometryObject*>::const_iterator obj;
  for (obj = d_geom_objs.begin(); obj != d_geom_objs.end(); ++obj) {  
    particleIndex count = 0;
    GeometryPiece* piece = (*obj)->getPiece();
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getBox();
    Box b = b1.intersect(b2);
    if(b.degenerate())
      count = 0;
    
    IntVector ppc = (*obj)->getNumParticlesPerCell();
    Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
    Vector dcorner = dxpp*0.5;
    // Size as a fraction of the cell size
    Vector size(1./((double) ppc.x()),
		1./((double) ppc.y()),
		1./((double) ppc.z()));
    

    SphereMembraneGeometryPiece* SMGP =
      dynamic_cast<SphereMembraneGeometryPiece*>(piece);
    if(SMGP){
      int numP = SMGP->createParticles(patch, position, pvolume,
				       pTang1, pTang2, pNorm, psize, start);
      for(int idx=0;idx<(start+numP);idx++){
	pvelocity[start+idx]=(*obj)->getInitialVelocity();
	ptemperature[start+idx]=(*obj)->getInitialTemperature();
       psp_vol[start+idx]=1.0/matl->getInitialDensity();
	pmass[start+idx]=matl->getInitialDensity() * pvolume[start+idx];
	// Determine if particle is on the surface
	pexternalforce[start+idx]=Vector(0,0,0); // for now
	IntVector cell_idx;
	if(patch->findCell(position[start+idx],cell_idx)){
	  long64 cellID = ((long64)cell_idx.x() << 16) |
	    ((long64)cell_idx.y() << 32) |
	    ((long64)cell_idx.z() << 48);
	  short int& myCellNAPID = cellNAPID[cell_idx];
	  ASSERT(myCellNAPID < 0x7fff);
	  myCellNAPID++;
	  pparticleID[start+idx] = cellID | (long64)myCellNAPID;
	}
	else{
	  cerr << "cellID is not right" << endl;
	}
      }
    }
    else{
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	Point lower = patch->nodePosition(*iter) + dcorner;
	for(int ix=0;ix < ppc.x(); ix++){
	  for(int iy=0;iy < ppc.y(); iy++){
	    for(int iz=0;iz < ppc.z(); iz++){
	      IntVector idx(ix, iy, iz);
	      Point p = lower + dxpp*idx;
	      IntVector cell_idx = *iter;
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
		pvelocity[start+count]=(*obj)->getInitialVelocity();
		ptemperature[start+count]=(*obj)->getInitialTemperature();
              psp_vol[start+count]     =1.0/matl->getInitialDensity();
                // Calculate particle mass
                double partMass = matl->getInitialDensity()*pvolume[start+count];
	        pmass[start+count] = partMass;

                // Apply the force BC if applicable
                Vector pExtForce(0,0,0);
                ParticleCreator::applyForceBC(dxpp, p, partMass, pExtForce);
	        pexternalforce[start+count] = pExtForce;

		// Determine if particle is on the surface
		psize[start+count] = size;
		pTang1[start+count] = Vector(1,0,0);
		pTang2[start+count] = Vector(0,0,1);
		pNorm[start+count]  = Vector(0,1,0);
		short int& myCellNAPID = cellNAPID[cell_idx];
		pparticleID[start+count] = cellID | (long64)myCellNAPID;
		ASSERT(myCellNAPID < 0x7fff);
		myCellNAPID++;
		
		count++;
		
	      }  // if inside
	    }  // loop in z
	  }  // loop in y
	}  // loop in x
      } // for
    } // else
    start += count; 
  }

  return subset;

}

particleIndex MembraneParticleCreator::countParticles(const Patch* patch,
						      vector<GeometryObject*>& d_geom_objs) const
{
  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex MembraneParticleCreator::countParticles(GeometryObject* obj,
						      const Patch* patch) const
{

  GeometryPiece* piece = obj->getPiece();
  
  SphereMembraneGeometryPiece* SMGP =
    dynamic_cast<SphereMembraneGeometryPiece*>(piece);
  
  if(SMGP){
    return SMGP->returnParticleCount(patch);
  } else {
    return ParticleCreator::countParticles(obj,patch); 
  }
  
}


void 
MembraneParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)

{
  particle_state.push_back(lb->pTang1Label);
  particle_state_preReloc.push_back(lb->pTang1Label_preReloc);

  particle_state.push_back(lb->pTang2Label);
  particle_state_preReloc.push_back(lb->pTang2Label_preReloc);

  particle_state.push_back(lb->pNormLabel);
  particle_state_preReloc.push_back(lb->pNormLabel_preReloc);
}
