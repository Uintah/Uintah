#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SmoothGeomPiece.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
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
                                 MPMFlags* flags, 
                                 SimulationStateP& /*sharedState*/)
{
  d_useLoadCurves = flags->d_useLoadCurves;
  d_with_color = flags->d_with_color;
  d_fracture = flags->d_fracture;
  d_ref_temp = flags->d_ref_temp; // for thermal stress 

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

    // Special case exception for SmoothGeomPieces and FileGeometryPieces
    SmoothGeomPiece *sgp = dynamic_cast<SmoothGeomPiece*>(piece);
    vector<double>* volumes = 0;
    vector<double>* temperatures = 0;
    vector<Vector>* pforces = 0;
    vector<Vector>* pfiberdirs = 0;
    if (sgp) volumes = sgp->getVolume();
    if (sgp) temperatures = sgp->getTemperature();
    if (sgp) pforces = sgp->getForces();
    if (sgp) pfiberdirs = sgp->getFiberDirs();

    // For getting particle volumes (if they exist)
    vector<double>::const_iterator voliter;
    geomvols::key_type volkey(patch,*obj);
    if (volumes) {
      if (!volumes->empty()) voliter = d_object_vols[volkey].begin();
    }

    // For getting particle temps (if they exist)
    vector<double>::const_iterator tempiter;
    geomvols::key_type tempkey(patch,*obj);
    if (temperatures) {
      if (!temperatures->empty()) tempiter = d_object_temps[tempkey].begin();
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
      //cerr << "Point["<<pidx<<"]="<<*itr<<" Cell = "<<cell_idx<<endl;
 
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID);
      
      if (volumes) {
        if (!volumes->empty()) {
          pvolume[pidx] = *voliter;
          pmass[pidx] = matl->getInitialDensity()*pvolume[pidx];
          ++voliter;
        }
      }

      if (temperatures) {
        if (!temperatures->empty()) {
          ptemperature[pidx] = *tempiter;
          ++tempiter;
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


// Get the LoadCurveID applicable for this material point
// WARNING : Should be called only once per particle during a simulation 
// because it updates the number of particles to which a BC is applied.
int ParticleCreator::getLoadCurveID(const Point& pp, const Vector& dxpp)
{
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        
    //cerr << " BC Type = " << bcs_type << endl;
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
void ParticleCreator::printPhysicalBCs()
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

void 
ParticleCreator::applyForceBC(const Vector& dxpp, 
                              const Point& pp,
                              const double& pMass, 
                              Vector& pExtForce)
{
  for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[i]->getType();
        
    //cerr << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Force") {
      ForceBC* bc = dynamic_cast<ForceBC*>
        (MPMPhysicalBCFactory::mpmPhysicalBCs[i]);

      Box bcBox;
      if (d_fracture)
        bcBox = Box(bc->getLowerRange(), bc->getUpperRange());
      else
        bcBox = Box(bc->getLowerRange()-dxpp,bc->getUpperRange()+dxpp);

      //cerr << "BC Box = " << bcBox << " Point = " << pp << endl;
      if(bcBox.contains(pp)) {
        pExtForce = bc->getForceDensity() * pMass;
        //cerr << "External Force on Particle = " << pExtForce 
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
  new_dw->allocateAndPut(position,       lb->pXLabel,             subset);
  new_dw->allocateAndPut(pvelocity,      lb->pVelocityLabel,      subset); 
  new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass,          lb->pMassLabel,          subset);
  new_dw->allocateAndPut(pvolume,        lb->pVolumeLabel,        subset);
  new_dw->allocateAndPut(ptemperature,   lb->pTemperatureLabel,   subset);
  new_dw->allocateAndPut(pparticleID,    lb->pParticleIDLabel,    subset);
  new_dw->allocateAndPut(psize,          lb->pSizeLabel,          subset);
  new_dw->allocateAndPut(pfiberdir,      lb->pFiberDirLabel,      subset); 
  new_dw->allocateAndPut(perosion,       lb->pErosionLabel,       subset); 
  // for thermal stress
  new_dw->allocateAndPut(ptempPrevious,  lb->pTempPreviousLabel,  subset); 
  if (d_useLoadCurves) {
    new_dw->allocateAndPut(pLoadCurveID,   lb->pLoadCurveIDLabel,   subset); 
  }
  new_dw->allocateAndPut(pdisp,          lb->pDispLabel,          subset);

  return subset;
}

void ParticleCreator::allocateVariablesAddRequires(Task* task, 
                                                   const MPMMaterial* ,
                                                   const PatchSet* ,
                                                   MPMLabel* lb) const
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,lb->pDispLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pXLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pMassLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pParticleIDLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pTemperatureLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pVelocityLabel, Ghost::None);
  task->requires(Task::NewDW,lb->pExtForceLabel_preReloc, Ghost::None);
  //task->requires(Task::OldDW,lb->pExternalForceLabel, Ghost::None);
  task->requires(Task::NewDW,lb->pVolumeDeformedLabel, Ghost::None);
  //task->requires(Task::OldDW,lb->pVolumeLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pErosionLabel, Ghost::None);
  task->requires(Task::OldDW,lb->pSizeLabel, Ghost::None);
  // for thermal stress
  task->requires(Task::OldDW,lb->pTempPreviousLabel, Ghost::None); 

  if (d_useLoadCurves)
    task->requires(Task::OldDW,lb->pLoadCurveIDLabel, Ghost::None);

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
  // for thermal stress
  constParticleVariable<double> o_tempPrevious; 
  
  new_dw->allocateTemporary(pdisp,addset);
  new_dw->allocateTemporary(position, addset);
  new_dw->allocateTemporary(pvelocity,addset); 
  new_dw->allocateTemporary(pexternalforce,addset);
  new_dw->allocateTemporary(pmass,addset);
  new_dw->allocateTemporary(pvolume,addset);
  new_dw->allocateTemporary(ptemperature,addset);
  new_dw->allocateTemporary(pparticleID,addset);
  new_dw->allocateTemporary(psize,addset);
  new_dw->allocateTemporary(pLoadCurveID,addset); 
  new_dw->allocateTemporary(perosion,addset); 
  // for thermal stress
  new_dw->allocateTemporary(ptempPrevious,addset); 

  old_dw->get(o_disp,lb->pDispLabel,delset);
  old_dw->get(o_position,lb->pXLabel,delset);
  old_dw->get(o_mass,lb->pMassLabel,delset);
  old_dw->get(o_particleID,lb->pParticleIDLabel,delset);
  old_dw->get(o_temperature,lb->pTemperatureLabel,delset);
  old_dw->get(o_velocity,lb->pVelocityLabel,delset);
  new_dw->get(o_external_force,lb->pExtForceLabel_preReloc,delset);
  //old_dw->get(o_external_force,lb->pExternalForceLabel,delset);
  new_dw->get(o_volume,lb->pVolumeDeformedLabel,delset);
  //old_dw->get(o_volume,lb->pVolumeLabel,delset);
  new_dw->get(o_erosion,lb->pErosionLabel_preReloc,delset);
  old_dw->get(o_size,lb->pSizeLabel,delset);
  if (d_useLoadCurves) 
    old_dw->get(o_loadcurve,lb->pLoadCurveIDLabel,delset);
  //for thermal stress
  old_dw->get(o_tempPrevious,lb->pTempPreviousLabel,delset);   

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pdisp[*n]=o_disp[*o];
    position[*n] = o_position[*o];
    pvelocity[*n]=o_velocity[*o];
    pexternalforce[*n]=o_external_force[*o];
    pmass[*n] = o_mass[*o];
    pvolume[*n] = o_volume[*o];
    ptemperature[*n]=o_temperature[*o];
    pparticleID[*n]=o_particleID[*o];
    perosion[*n]=o_erosion[*o];
    psize[*n]=o_size[*o];
    if (d_useLoadCurves) 
      pLoadCurveID[*n]=o_loadcurve[*o];
    // for thermal stress
    ptempPrevious[*n]=o_tempPrevious[*o];  
  }
  
  (*newState)[lb->pDispLabel]=pdisp.clone();
  (*newState)[lb->pXLabel] = position.clone();
  (*newState)[lb->pVelocityLabel]=pvelocity.clone();
  (*newState)[lb->pExternalForceLabel]=pexternalforce.clone();
  (*newState)[lb->pMassLabel]=pmass.clone();
  (*newState)[lb->pVolumeLabel]=pvolume.clone();
  (*newState)[lb->pTemperatureLabel]=ptemperature.clone();
  (*newState)[lb->pParticleIDLabel]=pparticleID.clone();
  (*newState)[lb->pErosionLabel]=perosion.clone();
  (*newState)[lb->pSizeLabel]=psize.clone();
  if (d_useLoadCurves) 
    (*newState)[lb->pLoadCurveIDLabel]=pLoadCurveID.clone();
  // for thermal stress
  (*newState)[lb->pTempPreviousLabel]=ptempPrevious.clone();  
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
            throw InternalError("Particle created outside of patch?", __FILE__, __LINE__);
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
  psize[i] = size;

  pvelocity[i] = (*obj)->getInitialVelocity();
  ptemperature[i] = (*obj)->getInitialTemperature();
  pmass[i] = matl->getInitialDensity()*pvolume[i];
  pdisp[i] = Vector(0.,0.,0.);
  // for thermal stress
  //ptempPrevious[i] = d_ref_temp; // This is incorrect T_n ~= T_ref
                                   // T_n is the temperature at time t_n
  // Assume that the correct d_ref_temp is specified in the input file
  ptempPrevious[i] = (d_ref_temp > 0.0) ? d_ref_temp : ptemperature[i];

  Vector pExtForce(0,0,0);
  ParticleCreator::applyForceBC(dxpp, p, pmass[i], pExtForce);
  pexternalforce[i] = pExtForce;
  pfiberdir[i] = matl->getConstitutiveModel()->getInitialFiberDir();
  perosion[i] = 1.0;

  ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
         && cell_idx.z() <= 0xffff);
  long64 cellID = ((long64)cell_idx.x() << 16) | 
    ((long64)cell_idx.y() << 32) | ((long64)cell_idx.z() << 48);
  short int& myCellNAPID = cellNAPID[cell_idx];
  pparticleID[i] = (cellID | (long64) myCellNAPID);
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
    vector<double>* temps = sgp->getTemperature();
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
        if (!temps->empty()) {
          double temp = temps->at(ii); 
          d_object_temps[volkey].push_back(temp);
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
    //sgp->deletePoints();
    //sgp->deleteVolume();
  } else {
    createPoints(patch,obj);
  }
  
  return (particleIndex) d_object_points[key].size();
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

  // for thermal stress
  particle_state.push_back(lb->pTempPreviousLabel);
  particle_state_preReloc.push_back(lb->pTempPreviousLabel_preReloc);
  
  particle_state.push_back(lb->pParticleIDLabel);
  particle_state_preReloc.push_back(lb->pParticleIDLabel_preReloc);
  
  particle_state.push_back(lb->pErosionLabel);
  particle_state_preReloc.push_back(lb->pErosionLabel_preReloc);

  if (d_with_color){
    particle_state.push_back(lb->pColorLabel);
    particle_state_preReloc.push_back(lb->pColorLabel_preReloc);
  }

  particle_state.push_back(lb->pSizeLabel);
  particle_state_preReloc.push_back(lb->pSizeLabel_preReloc);

  if (d_useLoadCurves) {
    particle_state.push_back(lb->pLoadCurveIDLabel);
    particle_state_preReloc.push_back(lb->pLoadCurveIDLabel_preReloc);
  }

  particle_state.push_back(lb->pDeformationMeasureLabel);
  particle_state.push_back(lb->pStressLabel);

  particle_state_preReloc.push_back(lb->pDeformationMeasureLabel_preReloc);
  particle_state_preReloc.push_back(lb->pStressLabel_preReloc);

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
