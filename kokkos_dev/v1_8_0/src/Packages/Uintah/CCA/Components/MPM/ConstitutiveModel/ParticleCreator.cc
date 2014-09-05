#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/FileGeometryPiece.h>

using namespace Uintah;
using std::vector;


ParticleCreator::ParticleCreator(MPMMaterial* matl, MPMLabel* lb,int n8or27) 
  : d_8or27(n8or27)
{
  registerPermanentParticleState(matl,lb);
}

ParticleCreator::~ParticleCreator()
{
}


ParticleSubset* 
ParticleCreator::createParticles(MPMMaterial* matl,
				 particleIndex numParticles,
				 CCVariable<short int>& cellNAPID,
				 const Patch* patch,DataWarehouse* new_dw,
				 MPMLabel* lb,
				 vector<GeometryObject*>& d_geom_objs)
{
  
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numParticles,dwi,lb,patch,new_dw);

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

    // Special case exception for FileGeometryPieces
    FileGeometryPiece* fgp = dynamic_cast<FileGeometryPiece*>(piece);
    if (fgp) {
      Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
      IntVector ppc = (*obj)->getNumParticlesPerCell();
      Vector size(1./((double) ppc.x()),
		  1./((double) ppc.y()),
		  1./((double) ppc.z()));
      vector<Point>::const_iterator itr;
      vector<Point>* points = fgp->getPoints();
      vector<double>* volumes = fgp->getVolume();
      int i = 0;
      for (itr = points->begin(); itr != points->end(); ++itr) {
	if (b2.contains(*itr)) {
	  position[start+count] = (*itr);
	  if (volumes->empty())
	    pvolume[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
	  else
	    pvolume[start+count] = (*volumes)[i];
	  pvelocity[start+count]=(*obj)->getInitialVelocity();
	  ptemperature[start+count]=(*obj)->getInitialTemperature();
	  pmass[start+count]=matl->getInitialDensity()*pvolume[start+count];
	  // Apply the force BC if applicable
	  Vector pExtForce(0,0,0);
	  applyForceBC(dxpp, *itr, pmass[start+count], pExtForce);
	  pexternalforce[start+count] = pExtForce;
	  
	  // Determine if particle is on the surface
	  IntVector cell_idx;
	  if(patch->findCell(position[start+count],cell_idx)){
	    long64 cellID = ((long64)cell_idx.x() << 16) |
	      ((long64)cell_idx.y() << 32) |
	      ((long64)cell_idx.z() << 48);
	    short int& myCellNAPID = cellNAPID[cell_idx];
	    ASSERT(myCellNAPID < 0x7fff);
	    myCellNAPID++;
	    pparticleID[start+count] = cellID | (long64)myCellNAPID;
	  }
	  psize[start+count] = size;
	  count++;
	}
	i++;
      }

    } else {
      IntVector ppc = (*obj)->getNumParticlesPerCell();
      Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
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
		pvelocity[start+count]=(*obj)->getInitialVelocity();
		ptemperature[start+count]=(*obj)->getInitialTemperature();
		
		// Calculate particle mass
		double partMass = matl->getInitialDensity()*pvolume[start+count];
		pmass[start+count] = partMass;
		
		// Apply the force BC if applicable
		Vector pExtForce(0,0,0);
		applyForceBC(dxpp, p, partMass, pExtForce);
		pexternalforce[start+count] = pExtForce;
		
		// Determine if particle is on the surface
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
    } // end of else
    start += count;
  }


  return subset;
}

void ParticleCreator::applyForceBC(const Vector& dxpp, 
                                   const Point& pp,
                                   const double& pMass, 
				   Vector& pExtForce)
{
  for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[i]->getType();
        
    //cout << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Force") {
      ForceBC* bc = dynamic_cast<ForceBC*>
			(MPMPhysicalBCFactory::mpmPhysicalBCs[i]);

      const Box bcBox(bc->getLowerRange()-dxpp, bc->getUpperRange()+dxpp);
      //cout << "BC Box = " << bcBox << endl;
          
      if(bcBox.contains(pp)) {
        pExtForce = bc->getForceDensity() * pMass;
        //cout << "External Force on Particle = " << pExtForce 
        //     << " Force Density = " << bc->getForceDensity() 
        //     << " Particle Mass = " << pMass << endl;
      }
    }
  }
}

ParticleSubset* 
ParticleCreator::allocateVariables(particleIndex numParticles, 
				   int dwi,MPMLabel* lb, 
				   const Patch* patch,
				   DataWarehouse* new_dw)
{

  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,
							patch);
  new_dw->allocateAndPut(position, lb->pXLabel, subset);
  new_dw->allocateAndPut(pvelocity, lb->pVelocityLabel, subset); 
  new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass, lb->pMassLabel, subset);
  new_dw->allocateAndPut(pvolume, lb->pVolumeLabel, subset);
  new_dw->allocateAndPut(ptemperature, lb->pTemperatureLabel, subset);
  new_dw->allocateAndPut(pparticleID, lb->pParticleIDLabel, subset);
  new_dw->allocateAndPut(psize, lb->pSizeLabel, subset);

  return subset;

}

particleIndex 
ParticleCreator::countParticles(const Patch* patch,
				vector<GeometryObject*>& d_geom_objs) const
{
  particleIndex sum = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=d_geom_objs.begin(); geom != d_geom_objs.end(); ++geom) 
    sum += countParticles(*geom,patch);

  return sum;

}


particleIndex 
ParticleCreator::countParticles(GeometryObject* obj, const Patch* patch) const
{
   GeometryPiece* piece = obj->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   if(b.degenerate())
      return 0;

   // Special case exception for FileGeometryPiece
   FileGeometryPiece* fgp = dynamic_cast<FileGeometryPiece*>(piece);
   if (fgp) {
     particleIndex fgp_count = 0;
     vector<Point>::const_iterator itr;
     vector<Point>* points = fgp->getPoints();
     for (itr = points->begin(); itr != points->end(); ++itr) {
       if (b2.contains(*itr))
	 fgp_count++;
     }
     return fgp_count;
   }


   IntVector ppc = obj->getNumParticlesPerCell();
   Vector dxpp = patch->dCell()/obj->getNumParticlesPerCell();
   Vector dcorner = dxpp*0.5;
   particleIndex count = 0;

   for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
     Point lower = patch->nodePosition(*iter) + dcorner;
     for(int ix=0;ix < ppc.x(); ix++){
       for(int iy=0;iy < ppc.y(); iy++){
	 for(int iz=0;iz < ppc.z(); iz++){
	   IntVector idx(ix, iy, iz);
	   Point p = lower + dxpp*idx;
	   if(!b2.contains(p))
	     throw InternalError("Particle created outside of patch?");
	   
	   if(piece->inside(p))
	     count++;
	 }
       }
     }
   }
   
   return count;


}

void ParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
						     MPMLabel* lb)
{
  particle_state.push_back(lb->pVelocityLabel);
  particle_state_preReloc.push_back(lb->pVelocityLabel_preReloc);

  particle_state.push_back(lb->pExternalForceLabel);
  particle_state_preReloc.push_back(lb->pExtForceLabel_preReloc);

  particle_state.push_back(lb->pMassLabel);
  particle_state_preReloc.push_back(lb->pMassLabel_preReloc);

  particle_state.push_back(lb->pVolumeLabel);
  particle_state_preReloc.push_back(lb->pVolumeLabel_preReloc);

  particle_state.push_back(lb->pTemperatureLabel);
  particle_state_preReloc.push_back(lb->pTemperatureLabel_preReloc);

  particle_state.push_back(lb->pParticleIDLabel);
  particle_state_preReloc.push_back(lb->pParticleIDLabel_preReloc);

  if (d_8or27 == 27) {
    particle_state.push_back(lb->pSizeLabel);
    particle_state_preReloc.push_back(lb->pSizeLabel_preReloc);
  }

  matl->getConstitutiveModel()->addParticleState(particle_state,
						 particle_state_preReloc);

}
