#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using std::vector;
using std::cerr;
using std::ofstream;

ParticleCreator::ParticleCreator(MPMMaterial* matl, 
                                 MPMLabel* lb,
                                 int n8or27, 
                                 bool haveLoadCurve,
				 bool doErosion) 
  : d_8or27(n8or27), d_useLoadCurves(haveLoadCurve), d_doErosion(doErosion)
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
    // Special case exception for FileGeometryPieces
    FileGeometryPiece* fgp = dynamic_cast<FileGeometryPiece*>(piece);
    vector<double>* volumes = 0;
    if (fgp)
      volumes = fgp->getVolume();
    
    int i = 0;
    vector<Point>::const_iterator itr;
    geompoints::key_type key(patch,*obj);
    for (itr=d_object_points[key].begin();itr!=d_object_points[key].end(); 
	 ++itr) {
      
      IntVector cell_idx;
      patch->findCell(*itr,cell_idx);
      
      particleIndex pidx = start+count;      
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,
			 cellNAPID);
      
      if (volumes)
	pvolume[pidx] = (*volumes)[i];
      
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
      i++;
    }
    start += count;
  }
  return subset;
}


// Get the LoadCurveID applicable for this material point
// WARNING : Should be called only once per particle during a simulation 
// because it updates the number of particles to which a BC is applied.
int
ParticleCreator::getLoadCurveID(const Point& pp, const Vector& dxpp)
{
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        
    //cout << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Pressure") {
      PressureBC* pbc = 
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (pbc->flagMaterialPoint(pp, dxpp)) {
        return pbc->loadCurveID(); 
      }
    }
  }
  return 0;
}

// Print MPM physical boundary condition information
void
ParticleCreator::printPhysicalBCs()
{
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {
      PressureBC* pbc = 
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      cerr << *pbc << endl;
    }
  }
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
#ifdef FRACTURE
      const Box bcBox(bc->getLowerRange(), bc->getUpperRange());
#else
      const Box bcBox(bc->getLowerRange()-dxpp, 
                      bc->getUpperRange()+dxpp);
#endif           
      //cout << "BC Box = " << bcBox << " Point = " << pp << endl;
      if(bcBox.contains(pp)) {
        pExtForce = bc->getForceDensity() * pMass;
        cout << "External Force on Particle = " << pExtForce 
             << " Force Density = " << bc->getForceDensity() 
             << " Particle Mass = " << pMass << endl;
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
  new_dw->allocateAndPut(position,       lb->pXLabel,             subset);
  new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      subset); 
  new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass,          lb->pMassLabel,          subset);
  new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        subset);
  new_dw->allocateAndPut(ptemperature,   lb->pTemperatureLabel,   subset);
  new_dw->allocateAndPut(pparticleID,    lb->pParticleIDLabel,    subset);
  new_dw->allocateAndPut(psize,          lb->pSizeLabel,          subset);
  new_dw->allocateAndPut(psp_vol,        lb->pSp_volLabel,        subset); 
  if (d_useLoadCurves) {
    new_dw->allocateAndPut(pLoadCurveID,   lb->pLoadCurveIDLabel,   subset); 
  }
  new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          subset);

  return subset;
}

void ParticleCreator::allocateVariablesAddRequires(Task* task, 
						   const MPMMaterial* matl,
						   const PatchSet* patch,
						   MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDispLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pXLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pVelocityLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pExternalForceLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pMassLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pVolumeLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pTemperatureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pSp_volLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pParticleIDLabel, Ghost::None);

  if (d_8or27 == 27)
    task->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);

  if (d_useLoadCurves)
    task->requires(Task::OldDW,lb->pLoadCurveIDLabel, Ghost::None);

  if (d_doErosion)
    task->requires(Task::OldDW,lb->pErosionLabel, Ghost::None);

}


void ParticleCreator::allocateVariablesAdd(MPMLabel* lb,DataWarehouse* new_dw,
					   ParticleSubset* addset,
					   map<const VarLabel*, ParticleVariableBase*>* newState,
					   ParticleSubset* delset,
					   DataWarehouse* old_dw)
{
  ParticleSubset::iterator n,o;

  constParticleVariable<Vector> o_disp;
  constParticleVariable<Point> o_position;
  constParticleVariable<Vector> o_velocity;
  constParticleVariable<Vector> o_external_force;
  constParticleVariable<double> o_mass;
  constParticleVariable<double> o_volume;
  constParticleVariable<double> o_temperature;
  constParticleVariable<double> o_sp_vol;
  constParticleVariable<long64> o_particleID;
  constParticleVariable<Vector> o_size;
  constParticleVariable<int> o_loadcurve;
  constParticleVariable<double> o_erosion;

  new_dw->allocateTemporary(pdisp,addset);
  new_dw->allocateTemporary(position, addset);
  new_dw->allocateTemporary(pvelocity,addset); 
  new_dw->allocateTemporary(pexternalforce,addset);
  new_dw->allocateTemporary(pmass,addset);
  new_dw->allocateTemporary(pvolume,addset);
  new_dw->allocateTemporary(ptemperature,addset);
  new_dw->allocateTemporary(psp_vol,addset); 
  new_dw->allocateTemporary(pparticleID,addset);
  new_dw->allocateTemporary(psize,addset);
  new_dw->allocateTemporary(pLoadCurveID,addset); 
  new_dw->allocateTemporary(perosion,addset); 


  old_dw->get(o_disp,lb->pDispLabel,delset);
  old_dw->get(o_position,lb->pXLabel,delset);
  old_dw->get(o_velocity,lb->pVelocityLabel,delset);
  old_dw->get(o_external_force,lb->pExternalForceLabel,delset);
  old_dw->get(o_mass,lb->pMassLabel,delset);
  old_dw->get(o_volume,lb->pVolumeLabel,delset);
  old_dw->get(o_temperature,lb->pTemperatureLabel,delset);
  old_dw->get(o_sp_vol,lb->pSp_volLabel,delset);
  old_dw->get(o_particleID,lb->pParticleIDLabel,delset);
  if (d_8or27 == 27) 
    old_dw->get(o_size,lb->pSizeLabel,delset);
  if (d_useLoadCurves) 
    old_dw->get(o_loadcurve,lb->pLoadCurveIDLabel,delset);
  if (d_doErosion) 
    old_dw->get(o_erosion,lb->pErosionLabel,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pdisp[*n]=o_disp[*o];
    position[*n] = o_position[*o];
    pvelocity[*n]=o_velocity[*o];
    pexternalforce[*n]=o_external_force[*o];
    pmass[*n] = o_mass[*o];
    pvolume[*n] = o_volume[*o];
    ptemperature[*n]=o_temperature[*o];
    psp_vol[*n]=o_sp_vol[*o];
    pparticleID[*n]=o_particleID[*o];
    if (d_8or27 == 27) 
      psize[*n]=o_size[*o];
    if (d_useLoadCurves) 
      pLoadCurveID[*n]=o_loadcurve[*o];
    if (d_doErosion) 
      perosion[*n]=o_erosion[*o];
  }
  
  (*newState)[lb->pDispLabel]=pdisp.clone();
  (*newState)[lb->pXLabel] = position.clone();
  (*newState)[lb->pVelocityLabel]=pvelocity.clone();
  (*newState)[lb->pExternalForceLabel]=pexternalforce.clone();
  (*newState)[lb->pMassLabel]=pmass.clone();
  (*newState)[lb->pVolumeLabel]=pvolume.clone();
  (*newState)[lb->pTemperatureLabel]=ptemperature.clone();
  (*newState)[lb->pSp_volLabel]=psp_vol.clone();
  (*newState)[lb->pParticleIDLabel]=pparticleID.clone();
  if (d_8or27 == 27) 
    (*newState)[lb->pSizeLabel]=psize.clone();
  if (d_useLoadCurves) 
    (*newState)[lb->pLoadCurveIDLabel]=pLoadCurveID.clone();
  if (d_doErosion) 
    (*newState)[lb->pErosionLabel]=perosion.clone();
}


void ParticleCreator::createPoints(const Patch* patch, GeometryObject* obj)
{
  geompoints::key_type key(patch,obj);
  GeometryPiece* piece = obj->getPiece();
  Box b2 = patch->getBox();
  IntVector ppc = obj->getNumParticlesPerCell();
  Vector dxpp = patch->dCell()/ppc;
  Vector dcorner = dxpp*0.5;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    Point lower = patch->nodePosition(*iter) + dcorner;
    for(int ix=0;ix < ppc.x(); ix++){
      for(int iy=0;iy < ppc.y(); iy++){
	for(int iz=0;iz < ppc.z(); iz++){
	  IntVector idx(ix, iy, iz);
	  Point p = lower + dxpp*idx;
	  if (!b2.contains(p))
	    throw InternalError("Particle created outside of patch?");
	  if (piece->inside(p)) 
	    d_object_points[key].push_back(p);
	}
      }
    }
  }

}


void 
ParticleCreator::initializeParticle(const Patch* patch,
				    vector<GeometryObject*>::const_iterator obj,
				    MPMMaterial* matl,
				    Point p,
				    IntVector cell_idx,
				    particleIndex i,
				    CCVariable<short int>& cellNAPID)
{
  IntVector ppc = (*obj)->getNumParticlesPerCell();
  Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
  Vector size(1./((double) ppc.x()),
	      1./((double) ppc.y()),
	      1./((double) ppc.z()));
  position[i] = p;
  pvolume[i] = dxpp.x()*dxpp.y()*dxpp.z();
  pvelocity[i] = (*obj)->getInitialVelocity();
  ptemperature[i] = (*obj)->getInitialTemperature();
  psp_vol[i] = 1./matl->getInitialDensity();
  pmass[i] = matl->getInitialDensity()*pvolume[i];
  psize[i] = size;
  pdisp[i] = Vector(0.,0.,0.);
  Vector pExtForce(0,0,0);
  ParticleCreator::applyForceBC(dxpp, p, pmass[i], pExtForce);
  pexternalforce[i] = pExtForce;
  ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
	 && cell_idx.z() <= 0xffff);
  long64 cellID = ((long64)cell_idx.x() << 16) | 
    ((long64)cell_idx.y() << 32) | ((long64)cell_idx.z() << 48);
  short int& myCellNAPID = cellNAPID[cell_idx];
  pparticleID[i] = cellID | (long64) myCellNAPID;
  ASSERT(myCellNAPID < 0x7fff);
  myCellNAPID++;
}


particleIndex 
ParticleCreator::countParticles(const Patch* patch,
				vector<GeometryObject*>& d_geom_objs)
{
  particleIndex sum = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=d_geom_objs.begin(); geom != d_geom_objs.end(); ++geom) 
    sum += countAndCreateParticles(patch,*geom);
  
  return sum;
  
}


particleIndex 
ParticleCreator::countAndCreateParticles(const Patch* patch, 
					 GeometryObject* obj)
{
  geompoints::key_type key(patch,obj);
  GeometryPiece* piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()){
    cout << "B.DEGENERATE" << endl;
    return 0;
  }
  
  // Special case exception for FileGeometryPiece
  FileGeometryPiece* fgp = dynamic_cast<FileGeometryPiece*>(piece);
  if (fgp)
    d_object_points[key] = *(fgp->getPoints());
  else
    createPoints(patch,obj);
  
  return d_object_points[key].size();
}

vector<const VarLabel* > ParticleCreator::returnParticleState()
{
  return particle_state;
}

void ParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
						     MPMLabel* lb)
{
  particle_state.push_back(lb->pDispLabel);
  particle_state_preReloc.push_back(lb->pDispLabel_preReloc);

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

  particle_state.push_back(lb->pSp_volLabel);
  particle_state_preReloc.push_back(lb->pSp_volLabel_preReloc); 
  
  particle_state.push_back(lb->pParticleIDLabel);
  particle_state_preReloc.push_back(lb->pParticleIDLabel_preReloc);

  if (d_8or27 == 27) {
    particle_state.push_back(lb->pSizeLabel);
    particle_state_preReloc.push_back(lb->pSizeLabel_preReloc);
  }

  if (d_useLoadCurves) {
    particle_state.push_back(lb->pLoadCurveIDLabel);
    particle_state_preReloc.push_back(lb->pLoadCurveIDLabel_preReloc);
  }

  if (d_doErosion) {
    particle_state.push_back(lb->pErosionLabel);
    particle_state_preReloc.push_back(lb->pErosionLabel_preReloc);
  }

  matl->getConstitutiveModel()->addParticleState(particle_state,
						 particle_state_preReloc);

}

int ParticleCreator::checkForSurface(const GeometryPiece* piece, const Point p,
		                     const Vector dxpp)
{

  //  Check the candidate points which surround the point just passed
  //   in.  If any of those points are not also inside the object
  //  the current point is on the surface
  
  int ss = 0;
  
  // Check to the left (-x)
  if(!piece->inside(p-Vector(dxpp.x(),0.,0.)))
    ss++;
  // Check to the right (+x)
  if(!piece->inside(p+Vector(dxpp.x(),0.,0.)))
    ss++;
  // Check behind (-y)
  if(!piece->inside(p-Vector(0.,dxpp.y(),0.)))
    ss++;
  // Check in front (+y)
  if(!piece->inside(p+Vector(0.,dxpp.y(),0.)))
    ss++;
#if 0
  // Check below (-z)
  if(!piece->inside(p-Vector(0.,0.,dxpp.z())))
    ss++;
  // Check above (+z)
  if(!piece->inside(p+Vector(0.,0.,dxpp.z())))
    ss++;
#endif
  
  if(ss>0){
    return 1;
  }
  else {
    return 0;
  }
}
