#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ExpTIParticleCreator.h>
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

ExpTIParticleCreator::ExpTIParticleCreator(MPMMaterial* matl,
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

ExpTIParticleCreator::~ExpTIParticleCreator()
{
}

ParticleSubset* 
ExpTIParticleCreator::createParticles(MPMMaterial* matl, 
					 particleIndex numParticles,
					 CCVariable<short int>& cellNAPID,
					 const Patch* patch,
					 DataWarehouse* new_dw,
					 MPMLabel* lb,
					 vector<GeometryObject*>& d_geom_objs)
{

  // Print the physical boundary conditions
  printPhysicalBCs();

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

    Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();    

    // Special case exception for SmoothGeomPieces and FileGeometryPieces
    SmoothGeomPiece *sgp = dynamic_cast<SmoothGeomPiece*>(piece);
    vector<double>* volumes = 0;
    vector<Vector>* pforces = 0;
    vector<Vector>* pfiberdirs = 0;
    if (sgp) volumes = sgp->getVolume();
    if (sgp) pforces = sgp->getForces();
    if (sgp) pfiberdirs = sgp->getFiberDirs();

    // For getting particle volumes (if they exist)
    vector<double>::const_iterator voliter;
    geomvols::key_type volkey(patch,*obj);
    if (volumes) {
      if (!volumes->empty()) voliter = d_object_vols[volkey].begin();
    }

    // For getting particle external forces (if they exist)
    vector<Vector>::const_iterator forceiter;
    geomvecs::key_type pforcekey(patch,*obj);
    if (pforces) {
      if (!pforces->empty()) forceiter = d_object_forces[pforcekey].begin();
    }

    // For getting particle fiber directions (if they exist)
    vector<Vector>::const_iterator fiberiter;
    geomvecs::key_type pfiberkey(patch,*obj);
    if (pfiberdirs) {
      if (!pfiberdirs->empty()) fiberiter = d_object_fibers[pfiberkey].begin();
    }

    vector<Point>::const_iterator itr;
    geompoints::key_type key(patch,*obj);
    for(itr=d_object_points[key].begin();itr!=d_object_points[key].end();++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;
      
      particleIndex pidx = start+count;      
 
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID);
      
      if (volumes) {
        if (!volumes->empty()) {
    	  pvolume[pidx] = *voliter;
          pmass[pidx] = matl->getInitialDensity()*pvolume[pidx];
          ++voliter;
        }
      }

      if (pforces) {
        if (!pforces->empty()) {
    	  pexternalforce[pidx] = *forceiter;
          ++forceiter;
        }
      }

      if (pfiberdirs) {
        if (!pfiberdirs->empty()) {
    	  pfiberdir[pidx] = *fiberiter;
          ++fiberiter;
        }
      }

      // If the particle is on the surface and if there is
      // a physical BC attached to it then mark with the 
      // physical BC pointer
      if (d_useLoadCurves) {
	if (checkForSurface(piece,*itr,dxpp)) {
	  pLoadCurveID[pidx] = getLoadCurveID(*itr, dxpp);
	} else {
	  pLoadCurveID[pidx] = 0;
	}
      }
      count++;
    }
    start += count;
  }

  return subset;
}

void 
ExpTIParticleCreator::initializeParticle(const Patch* patch,
				 vector<GeometryObject*>::const_iterator obj,
					 MPMMaterial* matl,
					 Point p, IntVector cell_idx,
					 particleIndex i,
					 CCVariable<short int>& cellNAPI)
{

  ParticleCreator::initializeParticle(patch,obj,matl,p,cell_idx,i,cellNAPI);

  pfiberdir[i] = Vector(0.,0.,1.);
}

particleIndex 
ExpTIParticleCreator::countParticles(const Patch* patch,
					vector<GeometryObject*>& d_geom_objs) 
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
ExpTIParticleCreator::countAndCreateParticles(const Patch* patch,
						 GeometryObject* obj) 
{
  geompoints::key_type key(patch,obj);
  geomvols::key_type volkey(patch,obj);
  geomvecs::key_type forcekey(patch,obj);
  geomvecs::key_type fiberkey(patch,obj);
  GeometryPiece* piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;
  
  // If the object is a SmoothGeomPiece (e.g. FileGeometryPiece or
  // SmoothCylGeomPiece) then use the particle creators in that 
  // class to do the counting
  SmoothGeomPiece   *sgp = dynamic_cast<SmoothGeomPiece*>(piece);
  if (sgp) {
    int numPts = 0;
    FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece);
    if(fgp){
      fgp->readPoints(patch->getID());
      numPts = fgp->returnPointCount();
    } else {
      Vector dxpp = patch->dCell()/obj->getNumParticlesPerCell();    
      double dx = Min(Min(dxpp.x(),dxpp.y()), dxpp.z());
      sgp->setParticleSpacing(dx);
      numPts = sgp->createPoints();
    }
    vector<Point>* points = sgp->getPoints();
    vector<double>* vols = sgp->getVolume();
    vector<Vector>* pforces = sgp->getForces();
    vector<Vector>* pfiberdirs = sgp->getFiberDirs();
    Point p;
    IntVector cell_idx;
    for (int ii = 0; ii < numPts; ++ii) {
      p = points->at(ii);
      if (patch->findCell(p,cell_idx)) {
        d_object_points[key].push_back(p);
        if (!vols->empty()) {
          double vol = vols->at(ii); 
          d_object_vols[volkey].push_back(vol);
        }
        if (!pforces->empty()) {
          Vector pforce = pforces->at(ii); 
          d_object_forces[forcekey].push_back(pforce);
        }
        if (!pfiberdirs->empty()) {
          Vector pfiber = pfiberdirs->at(ii); 
          d_object_fibers[fiberkey].push_back(pfiber);
        }
      }
    }
  } else {
    createPoints(patch,obj);
  }
  
  return d_object_points[key].size();
}


ParticleSubset* 
ExpTIParticleCreator::allocateVariables(particleIndex numParticles, 
					   int dwi,MPMLabel* lb, 
					   const Patch* patch,
					   DataWarehouse* new_dw)
{

  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
							      dwi,lb,patch,
							      new_dw);

  new_dw->allocateAndPut(pfiberdir, lb->pFiberDirLabel,    subset);

  return subset;

}

void
ExpTIParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
							MPMLabel* lb)
{
  particle_state.push_back(lb->pFiberDirLabel);
  particle_state_preReloc.push_back(lb->pFiberDirLabel_preReloc);
}
