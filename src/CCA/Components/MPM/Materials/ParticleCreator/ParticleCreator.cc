/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/HydroMPMLabel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/AMRMPMLabel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/TorqueBC.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/ArchesHeatFluxBC.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/MMS/MMS.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

#include <iostream>

/*  This code is a bit tough to follow.  Here's the basic order of operations.

First, MPM::actuallyInitialize calls MPMMaterial::createParticles, which in
turn calls ParticleCreator::createParticles for the appropriate ParticleCreator

Next,  createParticles, below, first loops over all of the geom_objects and
calls countAndCreateParticles.  countAndCreateParticles returns the number of
particles on a given patch associated with each geom_object and accumulates
that into a variable called num_particles.  countAndCreateParticles gets
the number of particles by calling createPoints, below.  When createPoints is
called, as each particle is determined to be inside of the object, it is pushed
back into the object_points entry of the ObjectVars struct.  ObjectVars
consists of several maps which are indexed on the GeometryObject and a vector
containing whatever data that entry is responsible for carrying.  A map is used
because even after particles are created, their initial data is still tied
back to the GeometryObject.  These might include velocity, temperature, color,
etc.
**** New in March 2023: With the addition of "recursive particle filling",
createPoints also computes particle volume and size, since those will change
with different levels of particle refinement.  ****

createPoints visits each cell,
and then depending on how many points are prescribed in the <res> tag in the
input file, loops over each of the candidate locations in that cell, and
determines if that point is inside or outside of the cell.  Points that are
inside the object are pushed back into the struct, as described above.  The
actual particle count comes from an operation in countAndCreateParticles
to determine the size of the object_points entry in the ObjectVars struct.

Now that we know how many particles we have for this material on this patch,
we are ready to allocateVariables, which calls allocateAndPut for all of the
variables needed in SerialMPM or AMRMPM.  At this point, storage for the
particles has been created, but the arrays allocated are still empty.

Now back in createParticles, the next step is to loop over all of the 
GeometryObjects.  Either way, loop over all of the particles in
object points and initialize the remaining particle data.  This is done 
for by calling initializeParticle.

initializeParticle, which is what is usually used, populates the particle data
based on what is specified in the <geometry_object> section of the
input file.  There is also an option to
call initializeParticlesForMMS, which is needed for running Method of
Manufactured Solutions, where special particle initialization is needed.

At that point, other than assigning particles to loadCurves, if called for,
we are done!

*/

using namespace Uintah;
using namespace std;

ParticleCreator::ParticleCreator(MPMMaterial* matl, 
                                 MPMFlags* flags)
{
  d_Hlb = scinew HydroMPMLabel();
  d_lb = scinew MPMLabel();
  d_useLoadCurves = flags->d_useLoadCurves;
  d_with_color = flags->d_with_color;
  d_artificial_viscosity = flags->d_artificial_viscosity;
  d_computeScaleFactor = flags->d_computeScaleFactor;
  d_doScalarDiffusion = flags->d_doScalarDiffusion;
  d_useCPTI = flags->d_useCPTI;

  d_flags = flags;

  // Hydro-mechanical coupling MPM
  d_coupledflow = flags->d_coupledflow;

  registerPermanentParticleState(matl);
}

ParticleCreator::~ParticleCreator()
{
  delete d_Hlb;
  delete d_lb;
}

particleIndex 
ParticleCreator::createParticles(MPMMaterial* matl,
                                 CCVariable<int>& cellNAPID,
                                 const Patch* patch,DataWarehouse* new_dw,
                                 vector<GeometryObject*>& geom_objs)
{
  ObjectVars vars;
  ParticleVars pvars;
  particleIndex numParticles = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=geom_objs.begin(); geom != geom_objs.end(); ++geom){ 
    numParticles += countAndCreateParticles(patch,*geom, vars);
  }

  int dwi = matl->getDWIndex();
  allocateVariables(numParticles,dwi,patch,new_dw, pvars);

  particleIndex start = 0;
  
  vector<GeometryObject*>::const_iterator obj;
  for (obj = geom_objs.begin(); obj != geom_objs.end(); ++obj) {
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

    // For getting particle volumes
    vector<double>::const_iterator voliter;
    voliter = vars.d_object_vols[*obj].begin();

    // For getting particle sizes
    vector<Matrix3>::const_iterator sizeiter;
    sizeiter = vars.d_object_size[*obj].begin();

    // Loop over all of the particles whose positions we know from
    // countAndCreateParticles, initialize the remaining variables
    vector<Point>::const_iterator itr;
    for(itr=vars.d_object_points[*obj].begin();
        itr!=vars.d_object_points[*obj].end(); ++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;

      if (!patch->containsPoint(*itr)) continue;
      
      particleIndex pidx = start+count;      

      // Use the volume and size computed at initialization
      pvars.pvolume[pidx] = *voliter;
      ++voliter;

      pvars.psize[pidx] = *sizeiter;
      ++sizeiter;

      // This initializes the remaining particle values
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID, pvars);

      // If the particle is on the surface and if there is
      // a physical BC attached to it then mark with the 
      // physical BC pointer
      if (d_useLoadCurves) {
        // if it is a surface particle
        if (pvars.psurface[pidx]==1) {
          Vector areacomps;
          pvars.pLoadCurveID[pidx] = getLoadCurveID(*itr, dxpp, areacomps, dwi);
          if (d_doScalarDiffusion) {
            pvars.parea[pidx]=Vector(pvars.parea[pidx].x()*areacomps.x(),
                                     pvars.parea[pidx].y()*areacomps.y(),
                                     pvars.parea[pidx].z()*areacomps.z());
          }
        } else {
          pvars.pLoadCurveID[pidx] = IntVector(0,0,0);
        }
        if(pvars.pLoadCurveID[pidx].x()==0 && d_doScalarDiffusion) {
          pvars.parea[pidx]=Vector(0.);
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
IntVector ParticleCreator::getLoadCurveID(const Point& pp, const Vector& dxpp,
                                          Vector& areacomps, int dwi)
{
  IntVector ret(0,0,0);
  int k=0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        
    //cerr << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Pressure") {
      PressureBC* pbc = 
        dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (pbc->flagMaterialPoint(pp, dxpp) 
       && (pbc->loadCurveMatl()==dwi || pbc->loadCurveMatl()==-99)) {
         ret(k) = pbc->loadCurveID();
         k++;
      }
    }
    else if (bcs_type == "Torque") {
      TorqueBC* tbc =
        dynamic_cast<TorqueBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (tbc->flagMaterialPoint(pp, dxpp)
       && (tbc->loadCurveMatl()==dwi || tbc->loadCurveMatl()==-99)) {
         ret(k) = tbc->loadCurveID();
         k++;
      }
    }
    else if (bcs_type == "ScalarFlux") {
      ScalarFluxBC* pbc = 
        dynamic_cast<ScalarFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (pbc->flagMaterialPoint(pp, dxpp, areacomps)) {
         ret(k) = pbc->loadCurveID(); 
         k++;
      }
    }
    else if (bcs_type == "HeatFlux") {      
      HeatFluxBC* hfbc = 
        dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (hfbc->flagMaterialPoint(pp, dxpp)
       && (hfbc->loadCurveMatl()==dwi || hfbc->loadCurveMatl()==-99)) {
         ret(k) = hfbc->loadCurveID(); 
         k++;
      }
    }
    else if (bcs_type == "ArchesHeatFlux") {      
      ArchesHeatFluxBC* hfbc = 
      dynamic_cast<ArchesHeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (hfbc->flagMaterialPoint(pp, dxpp)) {
         ret(k) = hfbc->loadCurveID(); 
         k++;
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
    if (bcs_type == "Torque") {
      TorqueBC* pbc =
        dynamic_cast<TorqueBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
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
  new_dw->allocateAndPut(pvars.pexternalhtrte,d_lb->pExternalHeatRateLabel,
                                                                        subset);
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
  new_dw->allocateAndPut(pvars.psurface,      d_lb->pSurfLabel,         subset);
  new_dw->allocateAndPut(pvars.psurfgrad,     d_lb->pSurfGradLabel,     subset);

  if(d_flags->d_integrator_type=="explicit"){
    new_dw->allocateAndPut(pvars.pvelGrad,    d_lb->pVelGradLabel,      subset);
  }
  new_dw->allocateAndPut(pvars.pTempGrad,   d_lb->pTemperatureGradientLabel,
                                                                        subset);
  if (d_useLoadCurves) {
    new_dw->allocateAndPut(pvars.pLoadCurveID,d_lb->pLoadCurveIDLabel,  subset);
  }
  if(d_with_color){
     new_dw->allocateAndPut(pvars.pcolor,     d_lb->pColorLabel,        subset);
  }
  if(d_doScalarDiffusion){
     new_dw->allocateAndPut(pvars.parea,  d_lb->diffusion->pArea,       subset);

     new_dw->allocateAndPut(pvars.pConcentration,
                                      d_lb->diffusion->pConcentration,  subset);
     new_dw->allocateAndPut(pvars.pConcPrevious,
                                      d_lb->diffusion->pConcPrevious,   subset);
     new_dw->allocateAndPut(pvars.pConcGrad,
                                 d_lb->diffusion->pGradConcentration,   subset);
     new_dw->allocateAndPut(pvars.pExternalScalarFlux,
                                  d_lb->diffusion->pExternalScalarFlux, subset);
  }

  if (d_coupledflow) {  // Harmless that rigid allocates and put, as long as
                        // nothing it put
      new_dw->allocateAndPut(pvars.pSolidMass, d_Hlb->pSolidMassLabel, subset);
      new_dw->allocateAndPut(pvars.pFluidMass, d_Hlb->pFluidMassLabel, subset);
      new_dw->allocateAndPut(pvars.pPorosity, d_Hlb->pPorosityLabel, subset);
      new_dw->allocateAndPut(pvars.pPorePressure, d_Hlb->pPorePressureLabel,
          subset);
      new_dw->allocateAndPut(pvars.pPrescribedPorePressure,
          d_Hlb->pPrescribedPorePressureLabel, subset);
      new_dw->allocateAndPut(pvars.pFluidVelocity, d_Hlb->pFluidVelocityLabel,
          subset);
  }

  if(d_artificial_viscosity){
     new_dw->allocateAndPut(pvars.p_q,        d_lb->p_qLabel,           subset);
  }
  if(d_flags->d_AMR){
     new_dw->allocateAndPut(pvars.pLastLevel, d_lb->pLastLevelLabel,    subset);
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
  Vector dxcc = patch->dCell();
  Vector dcorner = dxpp*0.5;
  int numLevelsParticleFilling =
                            obj->getInitialData_int("numLevelsParticleFilling");

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

  // AMR stuff
  const Level* curLevel = patch->getLevel();
  bool hasFiner = curLevel->hasFinerLevel();
  Level* fineLevel=0;
  if(hasFiner){
    fineLevel = (Level*) curLevel->getFinerLevel().get_rep();
  }

  Matrix3 stdSize(1./((double) ppc.x()),0.,0.,
                  0.,1./((double) ppc.y()),0.,
                  0.,0.,1./((double) ppc.z()));
  double c_vol = dxcc.x()*dxcc.y()*dxcc.z();

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    Point lower = patch->nodePosition(*iter) + dcorner;
    IntVector c = *iter;
    
    if(hasFiner){ // Don't create particles if a finer level exists here
      const Point CC = patch->cellPosition(c);
      bool includeExtraCells=false;
      const Patch* patchExists = fineLevel->getPatchFromPoint(CC,
                                                             includeExtraCells);
      if(patchExists != 0){
       continue;
      }
    }

    // Added on 3-18-23:  Ability to recursively add smaller particles to
    // fill in the gaps between a surface and the particles created using the
    // standard resolution.
    // The code immediately below does NOT fill recursively, it is essentially
    // just the original particle filling code
    if(abs(numLevelsParticleFilling)<=1){  // Original code
      for(int ix=0;ix < ppc.x(); ix++){
        for(int iy=0;iy < ppc.y(); iy++){
          for(int iz=0;iz < ppc.z(); iz++){

            IntVector idx(ix, iy, iz);
            Point p = lower + dxpp*idx;
            if (!b2.contains(p)){
              throw InternalError("Particle created outside of patch?",
                                   __FILE__, __LINE__);
            }
            if (piece->inside(p,true)){
              Vector p1(p(0),p(1),p(2));
              p1=affineTrans_A*p1+affineTrans_b;
              p(0)=p1[0];
              p(1)=p1[1];
              p(2)=p1[2];
              vars.d_object_points[obj].push_back(p);
              Matrix3 AS_size = affineTrans_A*stdSize;
              if(d_flags->d_axisymmetric){
                // assume unit radian extent in the circumferential direction
                double AS_vol = p.x()*(AS_size(0,0)*AS_size(1,1) -
                                       AS_size(0,1)*AS_size(1,0))*
                                       dxcc.x()*dxcc.y();
                vars.d_object_vols[obj].push_back(AS_vol);
              } else{
                vars.d_object_vols[obj].push_back(AS_size.Determinant()*c_vol);
              }
              vars.d_object_size[obj].push_back(AS_size);
            }
          }  // z
        }  // y
      }  // x
    } else {  // Do recursive particle filling
      // This code does recursive particle filling.  If the 
      // "numLevelsParticleFilling" variable, set in <geom_object>
      // is positive, then successively smaller particles are used to fill
      // in the empty space, but the originally created large particles may
      // stick out of the suface.  These are left alone.
      // If "numLevelsParticleFilling" is negative, then particles that
      // have a corner that falls outside the original surface are deleted
      // and replaced with sequentially smaller particles.

      int numInCell = 0;
      vector<Point> pointsInCell;
      vector<Vector> DXP;
      vector<double> pvolume;
      vector<Matrix3> psize;
      for(int ix=0;ix < ppc.x(); ix++){
        for(int iy=0;iy < ppc.y(); iy++){
          for(int iz=0;iz < ppc.z(); iz++){

            IntVector idx(ix, iy, iz);
            Point p = lower + dxpp*idx;
            if (!b2.contains(p)){
              throw InternalError("Particle created outside of patch?",
                                   __FILE__, __LINE__);
            }
            if (piece->inside(p,true)){
              Vector p1(p(0),p(1),p(2));
              p1=affineTrans_A*p1+affineTrans_b;
              p(0)=p1[0];
              p(1)=p1[1];
              p(2)=p1[2];
              pointsInCell.push_back(p);
              DXP.push_back(dxpp);
              Matrix3 AS_size = affineTrans_A*stdSize;
              if(d_flags->d_axisymmetric){
                // assume unit radian extent in the circumferential direction
                double AS_vol = p.x()*(AS_size(0,0)*AS_size(1,1) -
                                       AS_size(0,1)*AS_size(1,0))*
                                       dxcc.x()*dxcc.y();
                pvolume.push_back(AS_vol);
              } else{
                pvolume.push_back(AS_size.Determinant()*c_vol);
              }
              psize.push_back(AS_size);
              numInCell++;
            }
          }  // z
        }  // y
      }  // x

      Vector dxpr = dxpp;
      double mfactor = 1.;
      double dfactor = 1.;
      for (int rr = 1; rr < abs(numLevelsParticleFilling); rr++){
        int numPIC = pointsInCell.size();
        if(numLevelsParticleFilling < 0){  
          // Remove particles if a smaller particle within it would lie
          // outside the surface.  Fill them in below.
          vector<int> toRemove;
          toRemove.clear();
          for(int ip = 0; ip < numPIC; ip++){
             Point PIC  = pointsInCell[ip];
             Point corner[8];
             corner[0] = PIC + 0.25*Vector(-dxpr.x(),-dxpr.y(),- dxpr.z());
             corner[1] = PIC + 0.25*Vector(-dxpr.x(),-dxpr.y(),+ dxpr.z());
             corner[2] = PIC + 0.25*Vector(-dxpr.x(),+dxpr.y(),- dxpr.z());
             corner[3] = PIC + 0.25*Vector(-dxpr.x(),+dxpr.y(),+ dxpr.z());
             corner[4] = PIC + 0.25*Vector( dxpr.x(),-dxpr.y(),- dxpr.z());
             corner[5] = PIC + 0.25*Vector( dxpr.x(),-dxpr.y(),+ dxpr.z());
             corner[6] = PIC + 0.25*Vector( dxpr.x(),+dxpr.y(),- dxpr.z());
             corner[7] = PIC + 0.25*Vector( dxpr.x(),+dxpr.y(),+ dxpr.z());
             for(int ic = 0; ic < 8; ic++){
               if(!piece->inside(corner[ic],true)){
                 toRemove.push_back(ip);
                 break;
               }
             }
          }
          for(int ipo = toRemove.size()-1; ipo >= 0; ipo--){
            pointsInCell.erase(pointsInCell.begin() + toRemove[ipo]);
            DXP.erase(DXP.begin() + toRemove[ipo]);
            pvolume.erase(pvolume.begin() + toRemove[ipo]);
            psize.erase(psize.begin() + toRemove[ipo]);
            numInCell--;
          }
        }  // if numLevelsParticleFilling < 0
        numPIC = pointsInCell.size();
        dxpr*=0.5;
        Vector dcornerr = dxpr*0.5;
        mfactor*=2.;
        dfactor*=0.5;
        double dfCubed = dfactor*dfactor*dfactor;
        lower = patch->nodePosition(*iter) + dcornerr;
        for(int ix=0;ix < mfactor*ppc.x(); ix++){
          for(int iy=0;iy < mfactor*ppc.y(); iy++){
            for(int iz=0;iz < mfactor*ppc.z(); iz++){

              IntVector idx(ix, iy, iz);
              Point p = lower + dxpr*idx;
              if (!b2.contains(p)){
                throw InternalError("Particle created outside of patch?",
                                     __FILE__, __LINE__);
              }
              if (piece->inside(p,true)){ 
                Vector p1(p(0),p(1),p(2));
                p1=affineTrans_A*p1+affineTrans_b;
                p(0)=p1[0];
                p(1)=p1[1];
                p(2)=p1[2];
                bool overlap = false;
                for(int ip = 0; ip < numPIC; ip++){
                  Point PIC  = pointsInCell[ip];
                  Vector DXPip = DXP[ip];
                  if((p.x() >= PIC.x()-.5*DXPip.x()  && 
                      p.x() <= PIC.x()+.5*DXPip.x()) &&
                     (p.y() >= PIC.y()-.5*DXPip.y()  && 
                      p.y() <= PIC.y()+.5*DXPip.y()) &&
                     (p.z() >= PIC.z()-.5*DXPip.z()  && 
                      p.z() <= PIC.z()+.5*DXPip.z())) {
                      overlap = true;
                  }
                }
                if(!overlap){
                  pointsInCell.push_back(p);
                  DXP.push_back(dxpr);
                  Matrix3 AS_size = affineTrans_A*stdSize;
                  if(d_flags->d_axisymmetric){
                    // assume unit radian extent in  circumferential direction
                    double AS_vol = p.x()*(AS_size(0,0)*AS_size(1,1) -
                                           AS_size(0,1)*AS_size(1,0))*
                                           dxcc.x()*dxcc.y();
                    pvolume.push_back(dfactor*dfactor*AS_vol);
                  } else{
                    pvolume.push_back(AS_size.Determinant()*dfCubed*c_vol);
                  }
                  psize.push_back(dfactor*AS_size);
                  numInCell++;
                }
              }
            }  // z
          }  // y
        }  // x
      }  // for ... rr

      for(int ifc = 0; ifc<numInCell; ifc++){
        vars.d_object_points[obj].push_back(pointsInCell[ifc]);
        vars.d_object_vols[obj].push_back(pvolume[ifc]);
        vars.d_object_size[obj].push_back(psize[ifc]);
      }
    }  // do recursive particle filling
  }  // CellIterator

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
                                    CCVariable<int>& cellNAPID,
                                    ParticleVars& pvars)
{
  IntVector ppc = (*obj)->getInitialData_IntVector("res");
  Vector dxpp = patch->dCell()/(*obj)->getInitialData_IntVector("res");
  Vector dxcc = patch->dCell();

  Vector area(dxpp.y()*dxpp.z(),dxpp.x()*dxpp.z(),dxpp.x()*dxpp.y());

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

  // For AMR
  const Level* curLevel = patch->getLevel();
  pvars.prefined[i]     = curLevel->getIndex();

  //MMS
  string mms_type = d_flags->d_mms_type;
  if(!mms_type.empty()) {
   MMS MMSObject;
   Matrix3 size = pvars.psize[i];
   MMSObject.initializeParticleForMMS(pvars.position,pvars.pvelocity,
                                      pvars.psize,pvars.pdisp, pvars.pmass,
                                      pvars.pvolume,p,dxcc,size,patch,d_flags,i);
  } else {
    pvars.position[i] = p;
    pvars.pvelocity[i]  = (*obj)->getInitialData_Vector("velocity");
    if(d_flags->d_integrator_type=="explicit"){
      pvars.pvelGrad[i]  = Matrix3(0.0);
    }
    pvars.pTempGrad[i] = Vector(0.0);
  
    if (d_coupledflow &&
        !matl->getIsRigid()) {  // mass is determined by incoming porosity
        double rho_s = matl->getInitialDensity();
        double rho_w = matl->getWaterDensity();
        double n = matl->getPorosity();
        pvars.pmass[i] = (n * rho_w + (1.0 - n) * rho_s) * pvars.pvolume[i];
        pvars.pFluidMass[i] = rho_w * pvars.pvolume[i];
        pvars.pSolidMass[i] = rho_s * pvars.pvolume[i];
        pvars.pFluidVelocity[i] = pvars.pvelocity[i];
        pvars.pPorosity[i] = n;
        pvars.pPorePressure[i] = matl->getInitialPorepressure();
        pvars.pPrescribedPorePressure[i] = Vector(0., 0., 0.);
        pvars.pdisp[i] = Vector(0., 0., 0.);
    }
    else { // Using original line of MPM

      double vol_frac_CC = 1.0;
      try {
       if((*obj)->getInitialData_double("volumeFraction") == -1.0) {    
        vol_frac_CC = 1.0;
        pvars.pmass[i]      = matl->getInitialDensity()*pvars.pvolume[i];
       } else {
        vol_frac_CC = (*obj)->getInitialData_double("volumeFraction");
        pvars.pmass[i] = matl->getInitialDensity()*pvars.pvolume[i]*vol_frac_CC;
       }
      } catch (...) {
        vol_frac_CC = 1.0;       
        pvars.pmass[i]      = matl->getInitialDensity()*pvars.pvolume[i];
      }
      pvars.pdisp[i]        = Vector(0.,0.,0.);

    } // end else
  }
  
  if(d_with_color){
    pvars.pcolor[i] = (*obj)->getInitialData_double("color");
  }
  if(d_doScalarDiffusion){
    pvars.pConcentration[i] = (*obj)->getInitialData_double("concentration");
    pvars.pConcPrevious[i]  = pvars.pConcentration[i];
    pvars.pConcGrad[i]  = Vector(0.0);
    pvars.pExternalScalarFlux[i] = 0.0;
    pvars.parea[i]      = area;
  }
  if(d_artificial_viscosity){
    pvars.p_q[i] = 0.;
  }
  if(d_flags->d_AMR){
    pvars.pLastLevel[i] = curLevel->getID();
  }
  
  pvars.ptempPrevious[i]  = pvars.ptemperature[i];
  if(d_flags->d_useLogisticRegression ||
     d_useLoadCurves){
    GeometryPieceP piece = (*obj)->getPiece();
    pvars.psurface[i] = checkForSurface(piece,p,dxpp);
  } else {
    pvars.psurface[i] = 0.;
  }
  pvars.psurfgrad[i] = Vector(0.,0.,0.);

  pvars.pexternalforce[i] = Vector(0.,0.,0.);
  pvars.pexternalhtrte[i] = 0.;
  pvars.pfiberdir[i]      = matl->getConstitutiveModel()->getInitialFiberDir();

  ASSERT(cell_idx.x() <= 0xffff && 
         cell_idx.y() <= 0xffff && 
         cell_idx.z() <= 0xffff);
         
  long64 cellID = ((long64)cell_idx.x() << 16) | 
                  ((long64)cell_idx.y() << 32) | 
                  ((long64)cell_idx.z() << 48);
                  
  int& myCellNAPID = cellNAPID[cell_idx];
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
  
  createPoints(patch,obj,vars);
  
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

  if (d_flags->d_integrator_type == "explicit") {
    particle_state.push_back(d_lb->pExternalHeatRateLabel);
    particle_state_preReloc.push_back(d_lb->pExternalHeatRateLabel_preReloc);
  }

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

  if (d_doScalarDiffusion){
    particle_state.push_back(d_lb->diffusion->pConcentration);
    particle_state_preReloc.push_back(d_lb->diffusion->pConcentration_preReloc);

    particle_state.push_back(d_lb->diffusion->pConcPrevious);
    particle_state_preReloc.push_back(d_lb->diffusion->pConcPrevious_preReloc);

    particle_state.push_back(d_lb->diffusion->pGradConcentration);
    particle_state_preReloc.push_back(d_lb->diffusion->pGradConcentration_preReloc);

    particle_state.push_back(d_lb->diffusion->pExternalScalarFlux);
    particle_state_preReloc.push_back(d_lb->diffusion->pExternalScalarFlux_preReloc);

    particle_state.push_back(d_lb->diffusion->pArea);
    particle_state_preReloc.push_back(d_lb->diffusion->pArea_preReloc);

    matl->getScalarDiffusionModel()->addParticleState(particle_state,
                                                      particle_state_preReloc);
  }

  if (d_coupledflow && !matl->getIsRigid()) {
      //if (d_coupledflow ) {
      particle_state.push_back(d_Hlb->pFluidMassLabel);
      particle_state.push_back(d_Hlb->pSolidMassLabel);
      particle_state.push_back(d_Hlb->pPorePressureLabel);
      particle_state.push_back(d_Hlb->pPorosityLabel);

      // Error Cannot find in relocateParticles ???

      particle_state_preReloc.push_back(d_Hlb->pFluidMassLabel_preReloc);
      particle_state_preReloc.push_back(d_Hlb->pSolidMassLabel_preReloc);
      particle_state_preReloc.push_back(d_Hlb->pPorePressureLabel_preReloc);
      particle_state_preReloc.push_back(d_Hlb->pPorosityLabel_preReloc);

      if (d_flags->d_integrator_type == "explicit") {
        particle_state.push_back(d_Hlb->pFluidVelocityLabel);
        particle_state_preReloc.push_back(d_Hlb->pFluidVelocityLabel_preReloc);
      }
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

  if(!d_flags->d_AMR){
    particle_state.push_back(d_lb->pTemperatureGradientLabel);
    particle_state_preReloc.push_back(d_lb->pTemperatureGradientLabel_preReloc);
  }

  if (d_flags->d_refineParticles) {
    particle_state.push_back(d_lb->pRefinedLabel);
    particle_state_preReloc.push_back(d_lb->pRefinedLabel_preReloc);
  }

  particle_state.push_back(d_lb->pStressLabel);
  particle_state_preReloc.push_back(d_lb->pStressLabel_preReloc);

  particle_state.push_back(d_lb->pLocalizedMPMLabel);
  particle_state_preReloc.push_back(d_lb->pLocalizedMPMLabel_preReloc);

  if(d_flags->d_useLogisticRegression || d_flags->d_SingleFieldMPM){
    particle_state.push_back(d_lb->pSurfLabel);
    particle_state_preReloc.push_back(d_lb->pSurfLabel_preReloc);
  }

  if(d_flags->d_SingleFieldMPM){
    particle_state.push_back(d_lb->pSurfGradLabel);
    particle_state_preReloc.push_back(d_lb->pSurfGradLabel_preReloc);
  }

  if (d_artificial_viscosity) {
    particle_state.push_back(d_lb->p_qLabel);
    particle_state_preReloc.push_back(d_lb->p_qLabel_preReloc);
  }

  if (d_flags->d_AMR) {
    particle_state.push_back(d_lb->pLastLevelLabel);
    particle_state_preReloc.push_back(d_lb->pLastLevelLabel_preReloc);

  }

  if (d_computeScaleFactor) {
    particle_state.push_back(d_lb->pScaleFactorLabel);
    particle_state_preReloc.push_back(d_lb->pScaleFactorLabel_preReloc);
  }

  matl->getConstitutiveModel()->addParticleState(particle_state,
                                                 particle_state_preReloc);
                                                 
  matl->getDamageModel()->addParticleState( particle_state,
                                            particle_state_preReloc );
  
  matl->getErosionModel()->addParticleState( particle_state,
                                             particle_state_preReloc );
}

int
ParticleCreator::checkForSurface( const GeometryPieceP piece, const Point p,
                                  const Vector dxpp)
{

  //  Check the candidate points which surround the point just passed
  //   in.  If any of those points are not also inside the object
  //  the current point is on the surface
  
  int ss = 0;
  // Check to the left (-x)
  if(!piece->inside(p-Vector(dxpp.x(),0.,0.),true))
    ss++;
  // Check to the right (+x)
  if(!piece->inside(p+Vector(dxpp.x(),0.,0.),true))
    ss++;
  // Check behind (-y)
  if(!piece->inside(p-Vector(0.,dxpp.y(),0.),true))
    ss++;
  // Check in front (+y)
  if(!piece->inside(p+Vector(0.,dxpp.y(),0.),true))
    ss++;
  if (d_flags->d_ndim==3) {
    // Check below (-z)
    if(!piece->inside(p-Vector(0.,0.,dxpp.z()),true))
      ss++;
    // Check above (+z)
    if(!piece->inside(p+Vector(0.,0.,dxpp.z()),true))
      ss++;
  }

  if(ss>0){
    return 1;
  }
  else {
    return 0;
  }
}

double
ParticleCreator::checkForSurface2(const GeometryPieceP piece, const Point p,
                                  const Vector dxpp )
{

  //  Check the candidate points which surround the point just passed
  //  in.  If any of those points are not also inside the object
  //  the current point is on the surface
  
  int ss = 0;
  // Check to the left (-x)
  if(!piece->inside(p-Vector(dxpp.x(),0.,0.),true))
    ss++;
  // Check to the right (+x)
  if(!piece->inside(p+Vector(dxpp.x(),0.,0.),true))
    ss++;
  // Check behind (-y)
  if(!piece->inside(p-Vector(0.,dxpp.y(),0.),true))
    ss++;
  // Check in front (+y)
  if(!piece->inside(p+Vector(0.,dxpp.y(),0.),true))
    ss++;
  if (d_flags->d_ndim==3) {
    // Check below (-z)
    if(!piece->inside(p-Vector(0.,0.,dxpp.z()),true))
      ss++;
    // Check above (+z)
    if(!piece->inside(p+Vector(0.,0.,dxpp.z()),true))
      ss++;
  }

  if(ss>0){
    return 1.0;
  } else {
    return 0.0;
  }
#if 0
  else {
    // Check to the left (-x)
    if(!piece->inside(p-Vector(2.0*dxpp.x(),0.,0.),true))
      ss++;
    // Check to the right (+x)
    if(!piece->inside(p+Vector(2.0*dxpp.x(),0.,0.),true))
      ss++;
    // Check behind (-y)
    if(!piece->inside(p-Vector(0.,2.0*dxpp.y(),0.),true))
      ss++;
    // Check in front (+y)
    if(!piece->inside(p+Vector(0.,2.0*dxpp.y(),0.),true))
      ss++;
    // Check below (-z)
    if(!piece->inside(p-Vector(0.,0.,2.0*dxpp.z()),true))
      ss++;
    // Check above (+z)
    if(!piece->inside(p+Vector(0.,0.,2.0*dxpp.z()),true))
      ss++;
    // Check to the lower-left (-x,-y)
    if(!piece->inside(p-Vector(dxpp.x(),dxpp.y(),0.),true))
      ss++;
    // Check to the upper-right (+x,+y)
    if(!piece->inside(p+Vector(dxpp.x(),dxpp.y(),0.),true))
      ss++;
    // Check to the upper-left (-x,+z)
    if(!piece->inside(p+Vector(-dxpp.x(),dxpp.y(),0.),true))
      ss++;
    // Check to the lower-right (x,-z)
    if(!piece->inside(p+Vector(dxpp.x(),-dxpp.y(),0.),true))
      ss++;
    // Check to the lower-left (-x,-z)
    if(!piece->inside(p-Vector(dxpp.x(),0.,dxpp.z()),true))
      ss++;
    // Check to the upper-right (+x,+z)
    if(!piece->inside(p+Vector(dxpp.x(),0.,dxpp.z()),true))
      ss++;
    // Check to the upper-left (-x,+z)
    if(!piece->inside(p+Vector(-dxpp.x(),0.,dxpp.z()),true))
      ss++;
    // Check to the lower-right (x,-z)
    if(!piece->inside(p+Vector(dxpp.x(),0.,-dxpp.z()),true))
      ss++;
  }
  if(ss>0){
    return 0.0;
  } else {
    return 0.0;
  }
#endif
}
