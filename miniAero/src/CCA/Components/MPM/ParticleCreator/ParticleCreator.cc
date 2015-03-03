/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
using namespace std;

ParticleCreator::ParticleCreator(MPMMaterial* matl, 
                                 MPMFlags* flags)
:d_lock("Particle Creator lock")
{
  d_lb = scinew MPMLabel();
  d_useLoadCurves = flags->d_useLoadCurves;
  d_with_color = flags->d_with_color;
  d_artificial_viscosity = flags->d_artificial_viscosity;
  d_computeScaleFactor = flags->d_computeScaleFactor;
  d_useCPTI = flags->d_useCPTI;

  d_flags = flags;

  registerPermanentParticleState(matl);
}

ParticleCreator::~ParticleCreator()
{
  delete d_lb;
}

particleIndex 
ParticleCreator::createParticles(MPMMaterial* matl,
                                 CCVariable<short int>& cellNAPID,
                                 const Patch* patch,DataWarehouse* new_dw,
                                 vector<GeometryObject*>& d_geom_objs)
{
  ObjectVars vars;
  ParticleVars pvars;
  particleIndex numParticles = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=d_geom_objs.begin(); geom != d_geom_objs.end(); ++geom){ 
    numParticles += countAndCreateParticles(patch,*geom, vars);
  }
  
  int dwi = matl->getDWIndex();
  allocateVariables(numParticles,dwi,patch,new_dw, pvars);

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
    vector<Matrix3>* psizes       = 0;
    if (sgp){
      volumes      = sgp->getVolume();
      temperatures = sgp->getTemperature();
      pforces      = sgp->getForces();
      pfiberdirs   = sgp->getFiberDirs();
      pvelocities  = sgp->getVelocity();  // gcd adds and new change name
      psizes       = sgp->getSize();

      if(d_with_color){
        colors      = sgp->getColors();
      }
    }

    // For getting particle volumes (if they exist)
    vector<double>::const_iterator voliter;
    if (volumes) {
      if (!volumes->empty()) voliter = vars.d_object_vols[*obj].begin();
    }

    // For getting particle temps (if they exist)
    vector<double>::const_iterator tempiter;
    if (temperatures) {
      if (!temperatures->empty()) tempiter = vars.d_object_temps[*obj].begin();
    }

    // For getting particle external forces (if they exist)
    vector<Vector>::const_iterator forceiter;
    if (pforces) {
      if (!pforces->empty()) forceiter = vars.d_object_forces[*obj].begin();
    }

    // For getting particle fiber directions (if they exist)
    vector<Vector>::const_iterator fiberiter;
    if (pfiberdirs) {
      if (!pfiberdirs->empty()) fiberiter = vars.d_object_fibers[*obj].begin();
    }
    
    // For getting particle velocities (if they exist)   // gcd adds
    vector<Vector>::const_iterator velocityiter;
    if (pvelocities) {                             // new change name
      if (!pvelocities->empty()) velocityiter =
              vars.d_object_velocity[*obj].begin();  // new change name
    }                                                    // end gcd adds

    // For getting particle sizes (if they exist)
    vector<Matrix3>::const_iterator sizeiter;
    if (psizes) {
      if (!psizes->empty()) sizeiter = vars.d_object_size[*obj].begin();
    }

    // For getting particles colors (if they exist)
    vector<double>::const_iterator coloriter;
    if (colors) {
      if (!colors->empty()) coloriter = vars.d_object_colors[*obj].begin();
    }

    vector<Point>::const_iterator itr;
    for(itr=vars.d_object_points[*obj].begin();
        itr!=vars.d_object_points[*obj].end(); ++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;

      if (!patch->containsPoint(*itr)) continue;
      
      particleIndex pidx = start+count;      
 
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID, pvars);
      

      if (temperatures) {
        if (!temperatures->empty()) {
          pvars.ptemperature[pidx] = *tempiter;
          ++tempiter;
        }
      }

      if (pforces) {                           
        if (!pforces->empty()) {
          pvars.pexternalforce[pidx] = *forceiter;
          ++forceiter;
        }
      }

      if (pvelocities) {                           // gcd adds and change name 
        if (!pvelocities->empty()) {               // and change name
          pvars.pvelocity[pidx] = *velocityiter;
          ++velocityiter;
        }
      }                                         // end gcd adds

      if (pfiberdirs) {
        if (!pfiberdirs->empty()) {
          pvars.pfiberdir[pidx] = *fiberiter;
          ++fiberiter;
        }
      }

      if (volumes) {
        if (!volumes->empty()) {
          pvars.pvolume[pidx] = *voliter;
          pvars.pmass[pidx] = matl->getInitialDensity()*pvars.pvolume[pidx];
          ++voliter;
        }
      }

      // CPDI or CPTI
      if (psizes && ((d_flags->d_interpolator_type=="cpdi") || d_useCPTI)) {
        // Read psize from file
        if (!psizes->empty()) {
          Vector dxcc = patch->dCell(); 
          pvars.psize[pidx] = *sizeiter;
          if (volumes->empty()) {
            // Calculate CPDI hexahedron volume from psize 
            double volFactor=1.0;
            if (d_useCPTI) {
              // Calculate CPTI tetrahedron volume from psize 
              volFactor=6.0;
            }
            // (if volume not passed from FileGeometryPiece)
            pvars.pvolume[pidx]=abs(pvars.psize[pidx].Determinant()/volFactor);
            pvars.pmass[pidx] = matl->getInitialDensity()*pvars.pvolume[pidx];
          }
          // Modify psize (CPDI R-vectors) to be normalized by cell spacing
          Matrix3 size(1./((double) dxcc.x()),0.,0.,
                       0.,1./((double) dxcc.y()),0.,
                       0.,0.,1./((double) dxcc.z()));
          pvars.psize[pidx]= pvars.psize[pidx]*size;
          ++sizeiter;
        }
      }

      if (colors) {
        if (!colors->empty()) {
          pvars.pcolor[pidx] = *coloriter;
          ++coloriter;
        }
      }

      // If the particle is on the surface and if there is
      // a physical BC attached to it then mark with the 
      // physical BC pointer
      if (d_useLoadCurves) {
        if (checkForSurface(piece,*itr,dxpp)) {
          pvars.pLoadCurveID[pidx] = getLoadCurveID(*itr, dxpp);
        } else {
          pvars.pLoadCurveID[pidx] = 0;
        }
      }
      count++;
    }
    start += count;
  }
  return numParticles;
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
                                   DataWarehouse* new_dw,
                                   ParticleVars& pvars)
{
  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,
                                                        patch);
  new_dw->allocateAndPut(pvars.position,      d_lb->pXLabel,            subset);
  new_dw->allocateAndPut(pvars.pvelocity,     d_lb->pVelocityLabel,     subset); 
  new_dw->allocateAndPut(pvars.pexternalforce,d_lb->pExternalForceLabel,subset);
  new_dw->allocateAndPut(pvars.pmass,         d_lb->pMassLabel,         subset);
  new_dw->allocateAndPut(pvars.pvolume,       d_lb->pVolumeLabel,       subset);
  new_dw->allocateAndPut(pvars.ptemperature,  d_lb->pTemperatureLabel,  subset);
  new_dw->allocateAndPut(pvars.pparticleID,   d_lb->pParticleIDLabel,   subset);
  new_dw->allocateAndPut(pvars.psize,         d_lb->pSizeLabel,         subset);
  new_dw->allocateAndPut(pvars.plocalized,    d_lb->pLocalizedMPMLabel, subset);
  new_dw->allocateAndPut(pvars.prefined,      d_lb->pRefinedLabel,      subset);
  new_dw->allocateAndPut(pvars.pfiberdir,     d_lb->pFiberDirLabel,     subset);
  new_dw->allocateAndPut(pvars.ptempPrevious, d_lb->pTempPreviousLabel, subset);
  new_dw->allocateAndPut(pvars.pdisp,         d_lb->pDispLabel,         subset);
  if(d_flags->d_integrator_type=="explicit"){
    new_dw->allocateAndPut(pvars.pvelGrad,    d_lb->pVelGradLabel,      subset);
  }
  if (d_useLoadCurves) {
    new_dw->allocateAndPut(pvars.pLoadCurveID,d_lb->pLoadCurveIDLabel,  subset);
  }
  if(d_with_color){
     new_dw->allocateAndPut(pvars.pcolor,     d_lb->pColorLabel,        subset);
  }
  if(d_artificial_viscosity){
     new_dw->allocateAndPut(pvars.p_q,        d_lb->p_qLabel,           subset);
  }
  return subset;
}

void ParticleCreator::createPoints(const Patch* patch, GeometryObject* obj, 
                                                       ObjectVars& vars)
{

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
      const Patch* patchExists = fineLevel->getPatchFromPoint(CC,
                                                              includeExtraCells);
      if(patchExists != 0){
       continue;
      }
    }

    // Affine transformation for making conforming particle distributions
    // to be used in the conforming CPDI simulations. The input vectors are
    // optional and if you do not wish to use the affine transformation, just do
    // not define them in the input file.
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
            throw InternalError("Particle created outside of patch?",
                                 __FILE__, __LINE__);
          }
          if (piece->inside(p)){ 
            Vector p1(p(0),p(1),p(2));
            p1=affineTrans_A*p1+affineTrans_b;
            p(0)=p1[0];
            p(1)=p1[1];
            p(2)=p1[2];
            vars.d_object_points[obj].push_back(p);
          }
        }  // z
      }  // y
    }  // x
  }  // iterator

/*
//  This part is associated with CBDI_CompressiveCylinder.ups input file.
//  It creates conforming particle distribution to be used in the simulation.
//  To use that you need to uncomment the following commands to create the
//  conforming particle distribution and comment above commands that are used
//  to create non-conforming particle distributions.

  geompoints::key_type key(patch,obj);
  int resolutionPart=1;
  int nPar1=180*resolutionPart;
  int nPar2=16*resolutionPart;
    
  for(int ix=1;ix < nPar1+1; ix++){
    double ttemp,rtemp;
    ttemp=(ix-0.5)*2.0*3.14159265358979/nPar1;
    for(int iy=1;iy < nPar2+1; iy++){
        rtemp=0.75+(iy-0.5)*0.5/nPar2;
        Point p(rtemp*cos(ttemp),rtemp*sin(ttemp),0.5);
        if(patch->containsPoint(p)){
          d_object_points[key].push_back(p);
        }
    }
  }
*/

}


void 
ParticleCreator::initializeParticle(const Patch* patch,
                                    vector<GeometryObject*>::const_iterator obj,
                                    MPMMaterial* matl,
                                    Point p,
                                    IntVector cell_idx,
                                    particleIndex i,
                                    CCVariable<short int>& cellNAPID,
                                    ParticleVars& pvars)
{
  IntVector ppc = (*obj)->getInitialData_IntVector("res");
  Vector dxpp = patch->dCell()/(*obj)->getInitialData_IntVector("res");
  Vector dxcc = patch->dCell();

  // Affine transformation for making conforming particle distributions
  // to be used in the conforming CPDI simulations. The input vectors are
  // optional and if you do not wish to use the affine transformation, just do
  // not define them in the input file.

  Vector affineTrans_A0=(*obj)->getInitialData_Vector("affineTransformation_A0");
  Vector affineTrans_A1=(*obj)->getInitialData_Vector("affineTransformation_A1");
  Vector affineTrans_A2=(*obj)->getInitialData_Vector("affineTransformation_A2");
  Matrix3 affineTrans_A(
          affineTrans_A0[0],affineTrans_A0[1],affineTrans_A0[2],
          affineTrans_A1[0],affineTrans_A1[1],affineTrans_A1[2],
          affineTrans_A2[0],affineTrans_A2[1],affineTrans_A2[2]);
  // The size matrix is used for storing particle domain sizes (Rvectors for CPDI
  // and CPTI) normalized by the grid spacing
  Matrix3 size(1./((double) ppc.x()),0.,0.,
               0.,1./((double) ppc.y()),0.,
               0.,0.,1./((double) ppc.z()));
  size=affineTrans_A*size;
/*
// This part is associated with CBDI_CompressiveCylinder.ups input file.
// It determines particle domain sizes for the conforming particle distribution,
// which is used in the simulation.
// To activate that you need to uncomment the following commands to determine
// particle domain sizes for the conforming particle distribution, and
// comment above commands that are used to determine particle domain sizes for
// non-conforming particle distributions.

  int resolutionPart=1;
  int nPar1=180*resolutionPart;
  int nPar2=16*resolutionPart;
  double pi=3.14159265358979;
  double rtemp=sqrt(p.x()*p.x()+p.y()*p.y());
  Matrix3 size(2.*pi/nPar1*p.y()/dxcc[0],2.*0.25/nPar2/rtemp*p.x()/dxcc[1],0.,
              -2.*pi/nPar1*p.x()/dxcc[0],2.*0.25/nPar2/rtemp*p.y()/dxcc[1],0.,
                                      0.,                               0.,1.);
*/

  pvars.ptemperature[i] = (*obj)->getInitialData_double("temperature");
  pvars.plocalized[i]   = 0;
  pvars.prefined[i]     = 0;

  //MMS
  string mms_type = d_flags->d_mms_type;
  if(!mms_type.empty()) {
   MMS MMSObject;
   MMSObject.initializeParticleForMMS(pvars.position,pvars.pvelocity,
                                      pvars.psize,pvars.pdisp, pvars.pmass,
                                      pvars.pvolume,p,dxcc,size,patch,d_flags,i);
  } else {
    pvars.position[i] = p;
    if(d_flags->d_axisymmetric){
      // assume unit radian extent in the circumferential direction
      pvars.pvolume[i] = p.x()*
              (size(0,0)*size(1,1)-size(0,1)*size(1,0))*dxcc.x()*dxcc.y();
    } else {
      // standard voxel volume
      pvars.pvolume[i]  = size.Determinant()*dxcc.x()*dxcc.y()*dxcc.z();
    }

    pvars.psize[i]      = size;  // Normalized by grid spacing
   
    pvars.pvelocity[i]  = (*obj)->getInitialData_Vector("velocity");
    if(d_flags->d_integrator_type=="explicit"){
      pvars.pvelGrad[i] = Matrix3(0.0);
    }
  
    double vol_frac_CC = 1.0;
    try {
     if((*obj)->getInitialData_double("volumeFraction") == -1.0) {    
      vol_frac_CC = 1.0;
      pvars.pmass[i]      = matl->getInitialDensity()*pvars.pvolume[i];
     } else {
      vol_frac_CC = (*obj)->getInitialData_double("volumeFraction");
      pvars.pmass[i]   = matl->getInitialDensity()*pvars.pvolume[i]*vol_frac_CC;
     }
    } catch (...) {
      vol_frac_CC = 1.0;       
      pvars.pmass[i]      = matl->getInitialDensity()*pvars.pvolume[i];
    }
    pvars.pdisp[i]        = Vector(0.,0.,0.);
  }
  
  if(d_with_color){
    pvars.pcolor[i] = (*obj)->getInitialData_double("color");
  }
  if(d_artificial_viscosity){
    pvars.p_q[i] = 0.;
  }
  
  pvars.ptempPrevious[i]  = pvars.ptemperature[i];

  Vector pExtForce(0,0,0);
  applyForceBC(dxpp, p, pvars.pmass[i], pExtForce);
  
  pvars.pexternalforce[i] = pExtForce;
  pvars.pfiberdir[i]      = matl->getConstitutiveModel()->getInitialFiberDir();

  ASSERT(cell_idx.x() <= 0xffff && 
         cell_idx.y() <= 0xffff && 
         cell_idx.z() <= 0xffff);
         
  long64 cellID = ((long64)cell_idx.x() << 16) | 
                  ((long64)cell_idx.y() << 32) | 
                  ((long64)cell_idx.z() << 48);
                  
  short int& myCellNAPID = cellNAPID[cell_idx];
  pvars.pparticleID[i] = (cellID | (long64) myCellNAPID);
  ASSERT(myCellNAPID < 0x7fff);
  myCellNAPID++;
}


particleIndex 
ParticleCreator::countAndCreateParticles(const Patch* patch, 
                                         GeometryObject* obj,
                                         ObjectVars& vars)
{
  GeometryPieceP piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getExtraBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;
  
  // If the object is a SmoothGeomPiece (e.g. FileGeometryPiece or
  // SmoothCylGeomPiece) then use the particle creators in that 
  // class to do the counting
  SmoothGeomPiece   *sgp = dynamic_cast<SmoothGeomPiece*>(piece.get_rep());
  if (sgp) {
    int numPts = 0;
    FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece.get_rep());
    if(fgp){
      if (d_useCPTI) {
        proc0cout << "*** Reading CPTI file ***" << endl;
      }
      fgp->readPoints(patch->getID());
      numPts = fgp->returnPointCount();
    } else {
      Vector dxpp = patch->dCell()/obj->getInitialData_IntVector("res");    
      double dx   = Min(Min(dxpp.x(),dxpp.y()), dxpp.z());
      sgp->setParticleSpacing(dx);
      sgp->setCellSize(patch->dCell());
      numPts = sgp->createPoints();
    }
    vector<Point>* points      = sgp->getPoints();
    vector<double>* vols       = sgp->getVolume();
    vector<double>* temps      = sgp->getTemperature();
    vector<double>* colors     = sgp->getColors();
    vector<Vector>* pforces    = sgp->getForces();
    vector<Vector>* pfiberdirs = sgp->getFiberDirs();
    vector<Vector>* pvelocities= sgp->getVelocity();
    vector<Matrix3>* psizes    = sgp->getSize();
    Point p;
    IntVector cell_idx;
    
    //__________________________________
    // bulletproofing for smooth geometry pieces only
    BBox compDomain;
    GridP grid = patch->getLevel()->getGrid();
    grid->getSpatialRange(compDomain);

    Point min = compDomain.min();
    Point max = compDomain.max();
    for (int ii = 0; ii < numPts; ++ii) {
      p = points->at(ii);
      if(p.x() < min.x() || p.y() < min.y() || p.z() < min.z() ||
         p.x() > max.x() || p.y() > max.y() || p.z() > max.z() ){
        ostringstream warn;
        warn << "\n ERROR:MPM:ParticleCreator:SmoothGeometry Piece: the point ["
             << p << "] generated by this geometry piece "
             << " lies outside of the computational domain. \n" << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    //__________________________________
    //
    for (int ii = 0; ii < numPts; ++ii) {
      p = points->at(ii);
      if (patch->findCell(p,cell_idx)) {
        if (patch->containsPoint(p)) {
          vars.d_object_points[obj].push_back(p);
          
          if (!vols->empty()) {
            double vol = vols->at(ii); 
            vars.d_object_vols[obj].push_back(vol);
          }
          if (!temps->empty()) {
            double temp = temps->at(ii); 
            vars.d_object_temps[obj].push_back(temp);
          }
          if (!pforces->empty()) {
            Vector pforce = pforces->at(ii); 
            vars.d_object_forces[obj].push_back(pforce);
          }
          if (!pfiberdirs->empty()) {
            Vector pfiber = pfiberdirs->at(ii); 
            vars.d_object_fibers[obj].push_back(pfiber);
          }
          if (!pvelocities->empty()) {
            Vector pvel = pvelocities->at(ii); 
            vars.d_object_velocity[obj].push_back(pvel);
          }
          if (!psizes->empty()) {
            Matrix3 psz = psizes->at(ii); 
            vars.d_object_size[obj].push_back(psz);
          }
          if (!colors->empty()) {
            double color = colors->at(ii); 
            vars.d_object_colors[obj].push_back(color);
          }
        } 
      }  // patch contains cell
    }
    //sgp->deletePoints();
    //sgp->deleteVolume();
  } else {
    createPoints(patch,obj,vars);
  }
  
  return (particleIndex) vars.d_object_points[obj].size();
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

  if(d_flags->d_integrator_type=="explicit"){
    particle_state.push_back(d_lb->pVelGradLabel);
    particle_state_preReloc.push_back(d_lb->pVelGradLabel_preReloc);
  }

  particle_state.push_back(d_lb->pStressLabel);
  particle_state_preReloc.push_back(d_lb->pStressLabel_preReloc);

  particle_state.push_back(d_lb->pLocalizedMPMLabel);
  particle_state_preReloc.push_back(d_lb->pLocalizedMPMLabel_preReloc);

  if (d_flags->d_refineParticles) {
    particle_state.push_back(d_lb->pRefinedLabel);
    particle_state_preReloc.push_back(d_lb->pRefinedLabel_preReloc);
  }

  if (d_artificial_viscosity) {
    particle_state.push_back(d_lb->p_qLabel);
    particle_state_preReloc.push_back(d_lb->p_qLabel_preReloc);
  }

  if (d_computeScaleFactor) {
    particle_state.push_back(d_lb->pScaleFactorLabel);
    particle_state_preReloc.push_back(d_lb->pScaleFactorLabel_preReloc);
  }

  matl->getConstitutiveModel()->addParticleState(particle_state,
                                                 particle_state_preReloc);
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
