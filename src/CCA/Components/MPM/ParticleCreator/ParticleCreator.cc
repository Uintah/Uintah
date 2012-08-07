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


#include <CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/ArchesHeatFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <fstream>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::cerr;
using std::ofstream;

ParticleCreator::ParticleCreator(MPMMaterial* matl, 
                                 MPMFlags* flags)
:d_lock("Particle Creator lock")
{
  d_lb = scinew MPMLabel();
  d_useLoadCurves = flags->d_useLoadCurves;
  d_with_color = flags->d_with_color;
  d_artificial_viscosity = flags->d_artificial_viscosity;

  d_flags = flags;

  registerPermanentParticleState(matl);
}

ParticleCreator::~ParticleCreator()
{
  delete d_lb;
}

ParticleSubset* 
ParticleCreator::createParticles(MPMMaterial* matl,
                                 particleIndex numParticles,
                                 CCVariable<short int>& cellNAPID,
                                 const Patch* patch,DataWarehouse* new_dw,
                                 vector<GeometryObject*>& d_geom_objs)
{
  // Print the physical boundary conditions
  //  printPhysicalBCs();
  d_lock.writeLock();

  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numParticles,dwi,patch,new_dw);

  particleIndex start = 0;
  
  vector<GeometryObject*>::const_iterator obj;
  for (obj = d_geom_objs.begin(); obj != d_geom_objs.end(); ++obj) {
    particleIndex count = 0;
    GeometryPieceP piece = (*obj)->getPiece();
    Box b1 = piece->getBoundingBox();
    Box b2 = patch->getExtraBox();
    Box b = b1.intersect(b2);
    if(b.degenerate()) {
      count = 0;
      continue;
    }

    Vector dxpp = patch->dCell()/(*obj)->getInitialData_IntVector("res");    

    // Special case exception for SmoothGeomPieces and FileGeometryPieces
    SmoothGeomPiece *sgp = dynamic_cast<SmoothGeomPiece*>(piece.get_rep());
    vector<double>* volumes       = 0;
    vector<double>* temperatures  = 0;
    vector<double>* colors        = 0;
    vector<Vector>* pforces       = 0;
    vector<Vector>* pfiberdirs    = 0;
    vector<Vector>* pvelocities   = 0;    // gcd adds and new change name
    if (sgp){
      volumes      = sgp->getVolume();
      temperatures = sgp->getTemperature();
      pforces      = sgp->getForces();
      pfiberdirs   = sgp->getFiberDirs();
      pvelocities  = sgp->getVelocity();  // gcd adds and new change name

      if(d_with_color){
        colors      = sgp->getColors();
      }
    }

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
    
    // For getting particle velocities (if they exist)   // gcd adds
    vector<Vector>::const_iterator velocityiter;
    geomvecs::key_type pvelocitykey(patch,*obj);
    if (pvelocities) {                             // new change name
      if (!pvelocities->empty()) velocityiter =
              d_object_velocity[pvelocitykey].begin();  // new change name
    }                                                    // end gcd adds
    
    // For getting particles colors (if they exist)
    vector<double>::const_iterator coloriter;
    geomvols::key_type colorkey(patch,*obj);
    if (colors) {
      if (!colors->empty()) coloriter = d_object_colors[colorkey].begin();
    }

    vector<Point>::const_iterator itr;
    geompoints::key_type key(patch,*obj);
    for(itr=d_object_points[key].begin();itr!=d_object_points[key].end();++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;

      if (!patch->containsPoint(*itr)) continue;
      
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

      if (pvelocities) {                           // gcd adds and change name 
        if (!pvelocities->empty()) {               // and change name
          pvelocity[pidx] = *velocityiter;
          ++velocityiter;
        }
      }                                         // end gcd adds

      if (pfiberdirs) {
        if (!pfiberdirs->empty()) {
          pfiberdir[pidx] = *fiberiter;
          ++fiberiter;
        }
      }
      
      if (colors) {
        if (!colors->empty()) {
          pcolor[pidx] = *coloriter;
          ++coloriter;
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
  d_lock.writeUnlock();
  return subset;
}


// Get the LoadCurveID applicable for this material point
// WARNING : Should be called only once per particle during a simulation 
// because it updates the number of particles to which a BC is applied.
int ParticleCreator::getLoadCurveID(const Point& pp, const Vector& dxpp)
{
  int ret=0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        
    //cerr << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Pressure") {
      PressureBC* pbc = 
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (pbc->flagMaterialPoint(pp, dxpp)) {
         ret = pbc->loadCurveID(); 
      }
    }
    else if (bcs_type == "HeatFlux") {      
      HeatFluxBC* hfbc = 
        dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (hfbc->flagMaterialPoint(pp, dxpp)) {
        ret = hfbc->loadCurveID(); 
      }
    }
    else if (bcs_type == "ArchesHeatFlux") {      
      ArchesHeatFluxBC* hfbc = 
        dynamic_cast<ArchesHeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (hfbc->flagMaterialPoint(pp, dxpp)) {
        ret = hfbc->loadCurveID(); 
      }
    }
  }
  return ret;
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
    if (bcs_type == "HeatFlux") {
      HeatFluxBC* hfbc = 
        dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      cerr << *hfbc << endl;
    }
    if (bcs_type == "ArchesHeatFlux") {
      ArchesHeatFluxBC* hfbc = 
        dynamic_cast<ArchesHeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      cerr << *hfbc << endl;
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
                                   int dwi, const Patch* patch,
                                   DataWarehouse* new_dw)
{

  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,
                                                        patch);
  new_dw->allocateAndPut(position,       d_lb->pXLabel,             subset);
  new_dw->allocateAndPut(pvelocity,      d_lb->pVelocityLabel,      subset); 
  new_dw->allocateAndPut(pexternalforce, d_lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass,          d_lb->pMassLabel,          subset);
  new_dw->allocateAndPut(pvolume,        d_lb->pVolumeLabel,        subset);
  new_dw->allocateAndPut(ptemperature,   d_lb->pTemperatureLabel,   subset);
  new_dw->allocateAndPut(pparticleID,    d_lb->pParticleIDLabel,    subset);
  new_dw->allocateAndPut(psize,          d_lb->pSizeLabel,          subset);
  new_dw->allocateAndPut(pfiberdir,      d_lb->pFiberDirLabel,      subset); 
  // for thermal stress
  new_dw->allocateAndPut(ptempPrevious,  d_lb->pTempPreviousLabel,  subset); 
  new_dw->allocateAndPut(pdisp,          d_lb->pDispLabel,          subset);
  
  if (d_useLoadCurves) {
    new_dw->allocateAndPut(pLoadCurveID, d_lb->pLoadCurveIDLabel,   subset); 
  }
  if(d_with_color){
     new_dw->allocateAndPut(pcolor,      d_lb->pColorLabel,         subset);
  }
  if(d_artificial_viscosity){
     new_dw->allocateAndPut(p_q,      d_lb->p_qLabel,            subset);
  }
  return subset;
}

void ParticleCreator::allocateVariablesAddRequires(Task* task, 
                                                   const MPMMaterial* ,
                                                   const PatchSet* ) const
{
  d_lock.writeLock();
  Ghost::GhostType  gn = Ghost::None;
  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,d_lb->pDispLabel,        gn);
  task->requires(Task::OldDW,d_lb->pXLabel,           gn);
  task->requires(Task::OldDW,d_lb->pMassLabel,        gn);
  task->requires(Task::OldDW,d_lb->pParticleIDLabel,  gn);
  task->requires(Task::OldDW,d_lb->pTemperatureLabel, gn);
  task->requires(Task::OldDW,d_lb->pVelocityLabel,    gn);
  task->requires(Task::NewDW,d_lb->pExtForceLabel_preReloc, gn);
  //task->requires(Task::OldDW,d_lb->pExternalForceLabel,   gn);
  task->requires(Task::NewDW,d_lb->pVolumeLabel_preReloc,   gn);
  //task->requires(Task::OldDW,d_lb->pVolumeLabel,    gn);
  task->requires(Task::OldDW,d_lb->pSizeLabel,        gn);
  // for thermal stress
  task->requires(Task::OldDW,d_lb->pTempPreviousLabel, gn); 

  if (d_useLoadCurves){
    task->requires(Task::OldDW,d_lb->pLoadCurveIDLabel, gn);
  }
  if (d_with_color){
    task->requires(Task::OldDW,d_lb->pColorLabel,       gn);
  }
  if(d_artificial_viscosity){
    task->requires(Task::OldDW,d_lb->p_qLabel,          gn);
  }
  d_lock.writeUnlock();
}


void ParticleCreator::allocateVariablesAdd(DataWarehouse* new_dw,
                                           ParticleSubset* addset,
                                           map<const VarLabel*, ParticleVariableBase*>* newState,
                                           ParticleSubset* delset,
                                           DataWarehouse* old_dw)
{
  d_lock.writeLock();
  ParticleSubset::iterator n,o;

  constParticleVariable<Vector> o_disp;
  constParticleVariable<Point>  o_position;
  constParticleVariable<Vector> o_velocity;
  constParticleVariable<Vector> o_external_force;
  constParticleVariable<double> o_mass;
  constParticleVariable<double> o_volume;
  constParticleVariable<double> o_temperature;
  constParticleVariable<double> o_sp_vol;
  constParticleVariable<long64> o_particleID;
  constParticleVariable<Matrix3> o_size;
  constParticleVariable<int>    o_loadcurve;
  constParticleVariable<double> o_tempPrevious; // for thermal stress
  constParticleVariable<double> o_color;
  constParticleVariable<double> o_q;
  
  new_dw->allocateTemporary(pdisp,          addset);
  new_dw->allocateTemporary(position,       addset);
  new_dw->allocateTemporary(pvelocity,      addset); 
  new_dw->allocateTemporary(pexternalforce, addset);
  new_dw->allocateTemporary(pmass,          addset);
  new_dw->allocateTemporary(pvolume,        addset);
  new_dw->allocateTemporary(ptemperature,   addset);
  new_dw->allocateTemporary(pparticleID,    addset);
  new_dw->allocateTemporary(psize,          addset);
  new_dw->allocateTemporary(pLoadCurveID,   addset); 
  new_dw->allocateTemporary(ptempPrevious,  addset);

  old_dw->get(o_disp,           d_lb->pDispLabel,             delset);
  old_dw->get(o_position,       d_lb->pXLabel,                delset);
  old_dw->get(o_mass,           d_lb->pMassLabel,             delset);
  old_dw->get(o_particleID,     d_lb->pParticleIDLabel,       delset);
  old_dw->get(o_temperature,    d_lb->pTemperatureLabel,      delset);
  old_dw->get(o_velocity,       d_lb->pVelocityLabel,         delset);
  new_dw->get(o_external_force, d_lb->pExtForceLabel_preReloc,delset);
  //old_dw->get(o_external_force,d_lb->pExternalForceLabel,   delset);
  new_dw->get(o_volume,         d_lb->pVolumeLabel_preReloc,  delset);
  //old_dw->get(o_volume,       d_lb->pVolumeLabel,           delset);
  old_dw->get(o_size,           d_lb->pSizeLabel,             delset);
  old_dw->get(o_tempPrevious,   d_lb->pTempPreviousLabel,     delset);
  
  if (d_useLoadCurves){ 
    old_dw->get(o_loadcurve,    d_lb->pLoadCurveIDLabel,      delset);
  }
  if(d_with_color){
    new_dw->allocateTemporary(pcolor,         addset); 
    old_dw->get(o_color,        d_lb->pColorLabel,            delset);
  }
  if(d_artificial_viscosity){
    new_dw->allocateTemporary(p_q,         addset); 
    old_dw->get(o_q,        d_lb->p_qLabel,            delset);
  }
   

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pdisp[*n]         = o_disp[*o];
    position[*n]      = o_position[*o];
    pvelocity[*n]     = o_velocity[*o];
    pexternalforce[*n]= o_external_force[*o];
    pmass[*n]         = o_mass[*o];
    pvolume[*n]       = o_volume[*o];
    ptemperature[*n]  = o_temperature[*o];
    pparticleID[*n]   = o_particleID[*o];
    psize[*n]         = o_size[*o];
    ptempPrevious[*n] = o_tempPrevious[*o];  // for thermal stress
    if (d_useLoadCurves){ 
      pLoadCurveID[*n]= o_loadcurve[*o];
    }
    if (d_with_color){
      pcolor[*n]      = o_color[*o];
    }
    if(d_artificial_viscosity){
      p_q[*n]      = o_q[*o];
    }
  }

  (*newState)[d_lb->pDispLabel]           =pdisp.clone();
  (*newState)[d_lb->pXLabel]              =position.clone();
  (*newState)[d_lb->pVelocityLabel]       =pvelocity.clone();
  (*newState)[d_lb->pExternalForceLabel]  =pexternalforce.clone();
  (*newState)[d_lb->pMassLabel]           =pmass.clone();
  (*newState)[d_lb->pVolumeLabel]         =pvolume.clone();
  (*newState)[d_lb->pTemperatureLabel]    =ptemperature.clone();
  (*newState)[d_lb->pParticleIDLabel]     =pparticleID.clone();
  (*newState)[d_lb->pSizeLabel]           =psize.clone();
  (*newState)[d_lb->pTempPreviousLabel]   =ptempPrevious.clone(); // for thermal stress
  
  if (d_useLoadCurves){ 
    (*newState)[d_lb->pLoadCurveIDLabel]=pLoadCurveID.clone();
  }
  if(d_with_color){
    (*newState)[d_lb->pColorLabel]      =pcolor.clone();
  }
  if(d_artificial_viscosity){
    (*newState)[d_lb->p_qLabel]         =p_q.clone();
  }
  d_lock.writeUnlock();
}


void ParticleCreator::createPoints(const Patch* patch, GeometryObject* obj)
{
  geompoints::key_type key(patch,obj);
  GeometryPieceP piece = obj->getPiece();
  Box b2 = patch->getExtraBox();
  IntVector ppc = obj->getInitialData_IntVector("res");
  Vector dxpp = patch->dCell()/ppc;
  Vector dcorner = dxpp*0.5;

  // AMR stuff
  const Level* curLevel = patch->getLevel();
  bool hasFiner = curLevel->hasFinerLevel();
  Level* fineLevel=0;
  if(hasFiner){
    fineLevel = (Level*) curLevel->getFinerLevel().get_rep();
  }
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    Point lower = patch->nodePosition(*iter) + dcorner;
    IntVector c = *iter;
    
    if(hasFiner){ // Don't create particles if a finer level exists here
      const Point CC = patch->cellPosition(c);
      bool includeExtraCells=true;
      const Patch* patchExists = fineLevel->getPatchFromPoint(CC,includeExtraCells);
      if(patchExists != 0){
       continue;
      }
    }

    // Affine transformation for making conforming particle distributions
    //  to be used in the conforming CPDI simulations. The input vectors are
    //  optional and if you do not liketo use afine transformation, just do
    //  not define them in the input file.
    Vector affineTrans_A0=obj->getInitialData_Vector("affineTransformation_A0");
    Vector affineTrans_A1=obj->getInitialData_Vector("affineTransformation_A1");
    Vector affineTrans_A2=obj->getInitialData_Vector("affineTransformation_A2");
    Vector affineTrans_b= obj->getInitialData_Vector("affineTransformation_b");
    Matrix3 affineTrans_A(
            affineTrans_A0[0],affineTrans_A0[1],affineTrans_A0[2],
            affineTrans_A1[0],affineTrans_A1[1],affineTrans_A1[2],
            affineTrans_A2[0],affineTrans_A2[1],affineTrans_A2[2]);

    for(int ix=0;ix < ppc.x(); ix++){
      for(int iy=0;iy < ppc.y(); iy++){
        for(int iz=0;iz < ppc.z(); iz++){
        
          IntVector idx(ix, iy, iz);
          Point p = lower + dxpp*idx;
          if (!b2.contains(p)){
            throw InternalError("Particle created outside of patch?", __FILE__, __LINE__);
          }
          if (piece->inside(p)){ 
            Vector p1(p(0),p(1),p(2));
            p1=affineTrans_A*p1+affineTrans_b;
            p(0)=p1[0];
            p(1)=p1[1];
            p(2)=p1[2];
            d_object_points[key].push_back(p);
          }
        }  // z
      }  // y
    }  // x
  }  // iterator

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
  IntVector ppc = (*obj)->getInitialData_IntVector("res");
  Vector dxpp = patch->dCell()/(*obj)->getInitialData_IntVector("res");
  Vector dxcc = patch->dCell();

  // Affine transformation for making conforming particle distributions
  //  to be used in the conforming CPDI simulations. The input vectors are
  //  optional and if you do not liketo use afine transformation, just do
  //  not define them in the input file.
  Vector affineTrans_A0=(*obj)->getInitialData_Vector("affineTransformation_A0");
  Vector affineTrans_A1=(*obj)->getInitialData_Vector("affineTransformation_A1");
  Vector affineTrans_A2=(*obj)->getInitialData_Vector("affineTransformation_A2");
  Vector affineTrans_b= (*obj)->getInitialData_Vector("affineTransformation_b");
  Matrix3 affineTrans_A(
          affineTrans_A0[0],affineTrans_A0[1],affineTrans_A0[2],
          affineTrans_A1[0],affineTrans_A1[1],affineTrans_A1[2],
          affineTrans_A2[0],affineTrans_A2[1],affineTrans_A2[2]);
  Matrix3 size(1./((double) ppc.x()),0.,0.,
              0.,1./((double) ppc.y()),0.,
              0.,0.,1./((double) ppc.z()));

  size=affineTrans_A*size;
  ptemperature[i] = (*obj)->getInitialData_double("temperature");
//MMS
 string mms_type = d_flags->d_mms_type;
 if(!mms_type.empty()) {
	MMS MMSObject;
	MMSObject.initializeParticleForMMS(position,pvelocity,psize,pdisp,pmass,
						pvolume,p,dxcc,size,patch,d_flags,i);
 }  else {
	  position[i] = p;
	  if(d_flags->d_axisymmetric){
	    // assume unit radian extent in the circumferential direction
	    pvolume[i]  = p.x()*(size(0,0)*size(1,1)-size(0,1)*size(1,0))*dxcc.x()*dxcc.y();
	  } else {
	    // standard voxel volume
	    pvolume[i]  = size.Determinant()*dxcc.x()*dxcc.y()*dxcc.z();
	  }

	  psize[i]    = size;

	  pvelocity[i]    = (*obj)->getInitialData_Vector("velocity");

	  double vol_frac_CC = 1.0;
	  try {
	    if((*obj)->getInitialData_double("volumeFraction") == -1.0)
	    {    
	      vol_frac_CC = 1.0;
	      pmass[i]        = matl->getInitialDensity()*pvolume[i];
	    } else {
	      vol_frac_CC = (*obj)->getInitialData_double("volumeFraction");
	      pmass[i]        = matl->getInitialDensity()*pvolume[i]*vol_frac_CC;
	    }
	  } catch (...)
	  {
	    vol_frac_CC = 1.0;       
	    pmass[i]        = matl->getInitialDensity()*pvolume[i];
	  }
	  pdisp[i]        = Vector(0.,0.,0.);
}
  
  if(d_with_color){
    pcolor[i] = (*obj)->getInitialData_double("color");
  }
  if(d_artificial_viscosity){
    p_q[i] = 0.;
  }
  
  ptempPrevious[i]  = ptemperature[i];

  Vector pExtForce(0,0,0);
  applyForceBC(dxpp, p, pmass[i], pExtForce);
  
  pexternalforce[i] = pExtForce;
  pfiberdir[i]      = matl->getConstitutiveModel()->getInitialFiberDir();

  ASSERT(cell_idx.x() <= 0xffff && 
         cell_idx.y() <= 0xffff && 
         cell_idx.z() <= 0xffff);
         
  long64 cellID = ((long64)cell_idx.x() << 16) | 
                  ((long64)cell_idx.y() << 32) | 
                  ((long64)cell_idx.z() << 48);
                  
  short int& myCellNAPID = cellNAPID[cell_idx];
  pparticleID[i] = (cellID | (long64) myCellNAPID);
  ASSERT(myCellNAPID < 0x7fff);
  myCellNAPID++;
}

particleIndex 
ParticleCreator::countParticles(const Patch* patch,
                                vector<GeometryObject*>& d_geom_objs)
{
  d_lock.writeLock();
  particleIndex sum = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=d_geom_objs.begin(); geom != d_geom_objs.end(); ++geom){ 
    sum += countAndCreateParticles(patch,*geom);
  }
  
  d_lock.writeUnlock();
  return sum;
}


particleIndex 
ParticleCreator::countAndCreateParticles(const Patch* patch, 
                                         GeometryObject* obj)
{
  geompoints::key_type key(patch,obj);
  geomvols::key_type   volkey(patch,obj);
  geomvecs::key_type   forcekey(patch,obj);
  geomvecs::key_type   fiberkey(patch,obj);
  geomvecs::key_type   pvelocitykey(patch,obj);
  GeometryPieceP piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getExtraBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;
  
  // If the object is a SmoothGeomPiece (e.g. FileGeometryPiece or
  // SmoothCylGeomPiece) then use the particle creators in that 
  // class to do the counting d
  SmoothGeomPiece   *sgp = dynamic_cast<SmoothGeomPiece*>(piece.get_rep());
  if (sgp) {
    int numPts = 0;
    FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece.get_rep());
    if(fgp){
      fgp->readPoints(patch->getID());
      numPts = fgp->returnPointCount();
    } else {
      Vector dxpp = patch->dCell()/obj->getInitialData_IntVector("res");    
      double dx   = Min(Min(dxpp.x(),dxpp.y()), dxpp.z());
      sgp->setParticleSpacing(dx);
      numPts = sgp->createPoints();
    }
    vector<Point>* points      = sgp->getPoints();
    vector<double>* vols       = sgp->getVolume();
    vector<double>* temps      = sgp->getTemperature();
    vector<double>* colors     = sgp->getColors();
    vector<Vector>* pforces    = sgp->getForces();
    vector<Vector>* pfiberdirs = sgp->getFiberDirs();
    vector<Vector>* pvelocities= sgp->getVelocity();
    Point p;
    IntVector cell_idx;
    
    for (int ii = 0; ii < numPts; ++ii) {
      p = points->at(ii);
      if (patch->findCell(p,cell_idx)) {
        if (patch->containsPoint(p)) {
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
          if (!pvelocities->empty()) {
            Vector pvel = pvelocities->at(ii); 
            d_object_velocity[pvelocitykey].push_back(pvel);
          }
          if (!colors->empty()) {
            double color = colors->at(ii); 
            d_object_colors[volkey].push_back(color);
          }
        } 
      }  // patch contains cell
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


vector<const VarLabel* > ParticleCreator::returnParticleStatePreReloc()
{
  return particle_state_preReloc;
}

void ParticleCreator::registerPermanentParticleState(MPMMaterial* matl)
{
  d_lock.writeLock();
  particle_state.push_back(d_lb->pDispLabel);
  particle_state_preReloc.push_back(d_lb->pDispLabel_preReloc);

  particle_state.push_back(d_lb->pVelocityLabel);
  particle_state_preReloc.push_back(d_lb->pVelocityLabel_preReloc);

  particle_state.push_back(d_lb->pExternalForceLabel);
  particle_state_preReloc.push_back(d_lb->pExtForceLabel_preReloc);

  particle_state.push_back(d_lb->pMassLabel);
  particle_state_preReloc.push_back(d_lb->pMassLabel_preReloc);

  particle_state.push_back(d_lb->pVolumeLabel);
  particle_state_preReloc.push_back(d_lb->pVolumeLabel_preReloc);

  particle_state.push_back(d_lb->pTemperatureLabel);
  particle_state_preReloc.push_back(d_lb->pTemperatureLabel_preReloc);

  // for thermal stress
  particle_state.push_back(d_lb->pTempPreviousLabel);
  particle_state_preReloc.push_back(d_lb->pTempPreviousLabel_preReloc);
  
  particle_state.push_back(d_lb->pParticleIDLabel);
  particle_state_preReloc.push_back(d_lb->pParticleIDLabel_preReloc);
  

  if (d_with_color){
    particle_state.push_back(d_lb->pColorLabel);
    particle_state_preReloc.push_back(d_lb->pColorLabel_preReloc);
  }

  particle_state.push_back(d_lb->pSizeLabel);
  particle_state_preReloc.push_back(d_lb->pSizeLabel_preReloc);

  if (d_useLoadCurves) {
    particle_state.push_back(d_lb->pLoadCurveIDLabel);
    particle_state_preReloc.push_back(d_lb->pLoadCurveIDLabel_preReloc);
  }

  particle_state.push_back(d_lb->pDeformationMeasureLabel);
  particle_state_preReloc.push_back(d_lb->pDeformationMeasureLabel_preReloc);

  particle_state.push_back(d_lb->pStressLabel);
  particle_state_preReloc.push_back(d_lb->pStressLabel_preReloc);

  particle_state.push_back(d_lb->pdTdtLabel);
  particle_state_preReloc.push_back(d_lb->pdTdtLabel_preReloc);

  if (d_artificial_viscosity) {
    particle_state.push_back(d_lb->p_qLabel);
    particle_state_preReloc.push_back(d_lb->p_qLabel_preReloc);
  }

  matl->getConstitutiveModel()->addParticleState(particle_state,
                                                 particle_state_preReloc);
  d_lock.writeUnlock();
}

int
ParticleCreator::checkForSurface( const GeometryPieceP piece, const Point p,
                                  const Vector dxpp )
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
#if 1
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
