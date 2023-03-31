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

#include <CCA/Components/MPM/Materials/ParticleCreator/FileGeomPieceParticleCreator.h>
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

#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

#include <iostream>

/*  This code has been specialized to only handle the "file" and "smooth"
geometry objects.  File is inherited from Smooth, all of which is in
Core/GeometryPiece.

This code is a bit tough to follow.  Here's the basic order of operations.

First, MPM::actuallyInitialize calls MPMMaterial::createParticles, which in
turn calls ParticleCreator::createParticles for the appropriate ParticleCreator

Next,  createParticles, below, first loops over all of the geom_objects and
calls countAndCreateParticles.  countAndCreateParticles returns the number of
particles on a given patch associated with each geom_object and accumulates
that into a variable called num_particles.  countAndCreateParticles gets
the number of particles by querying the functions for smooth geometry 
piece types. 
For each particle that is determined to be inside of the object, it is pushed
back into the object_points entry of the ObjectVars struct.  ObjectVars
consists of several maps which are indexed on the GeometryObject and a vector
containing whatever data that entry is responsible for carrying.  A map is used
because even after particles are created, their initial data is still tied
back to the GeometryObject.  These might include velocity, temperature, color,
etc.  HOWEVER, it is also possible for a user to specify any number of data
entries on a per-particle basis, (e.g., p.volume) as long as this is indicated
in the input file.  see, e.g., inputs/MPM/cpti_disks.ups for an example

Now that we know how many particles we have for this material on this patch,
we are ready to allocateVariables, which calls allocateAndPut for all of the
variables needed in SerialMPM or AMRMPM.  At this point, storage for the
particles has been created, but the arrays allocated are still empty.

Now back in createParticles, the next step is to loop over all of the 
GeometryObjects.  SmoothGeometryPiece objects,
MAY have their own methods for populating the data .
Either way, loop over all of the particles in
object points and initialize the remaining particle data.  For the Smooth/File
pieces, if arrays exist that contain other data, use that data to populate the
other entries.

initializeParticle, which is what is usually used, populates the particle data
based on either what is specified in the <geometry_object> section of the
input file, or by geometric considerations (such as size, from which we get
volume, from which we get mass (volume*density).  There is also an option to
call initializeParticlesForMMS, which is needed for running Method of
Manufactured Solutions, where special particle initialization is needed.)

At that point, other than assigning particles to loadCurves, if called for,
we are done!

*/

using namespace Uintah;
using namespace std;

FileGeomPieceParticleCreator::FileGeomPieceParticleCreator(MPMMaterial* matl, 
                                                           MPMFlags* flags)
                                                  :  ParticleCreator(matl,flags)
{
}

FileGeomPieceParticleCreator::~FileGeomPieceParticleCreator()
{
}

particleIndex 
FileGeomPieceParticleCreator::createParticles(MPMMaterial* matl,
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

    // For SmoothGeomPieces and FileGeometryPieces
    SmoothGeomPiece *sgp = dynamic_cast<SmoothGeomPiece*>(piece.get_rep());
    vector<double>* volumes        = 0;
    vector<Matrix3>* psizes        = 0;
    vector<double>* temperatures   = 0;
    vector<double>* colors         = 0;
    vector<double>* concentrations = 0;
    vector<Vector>* pforces        = 0;
    vector<Vector>* pfiberdirs     = 0;
    vector<Vector>* pvelocities    = 0;
    vector<Vector>* pareas        = 0;

    volumes      = sgp->getVolume();
    temperatures = sgp->getTemperature();
    pforces      = sgp->getForces();
    pfiberdirs   = sgp->getFiberDirs();
    pvelocities  = sgp->getVelocity();  // gcd adds and new change name
    psizes       = sgp->getSize();

    if(d_with_color){
      colors      = sgp->getColors();
    }

    if(d_doScalarDiffusion){
      concentrations  = sgp->getConcentration();
      pareas          = sgp->getArea();
    }

    // For getting particle volumes (if they exist)
    vector<double>::const_iterator voliter;
    if (volumes) {
    voliter = vars.d_object_vols[*obj].begin();
      if (!volumes->empty()) voliter = vars.d_object_vols[*obj].begin();
    }

    // For getting particle sizes (if they exist)
    vector<Matrix3>::const_iterator sizeiter;
    if (psizes) {
      if (!psizes->empty()) sizeiter = vars.d_object_size[*obj].begin();
    sizeiter = vars.d_object_size[*obj].begin();
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

    // For getting particle areas (if they exist)
    vector<Vector>::const_iterator areaiter;
    if (pareas) {
      if (!pareas->empty()) areaiter = vars.d_object_area[*obj].begin();
    }

    // For getting particles colors (if they exist)
    vector<double>::const_iterator coloriter;
    if (colors) {
      if (!colors->empty()) coloriter = vars.d_object_colors[*obj].begin();
    }

    // For getting particles concentrations (if they exist)
    vector<double>::const_iterator concentrationiter;
    if (concentrations) {
      if (!concentrations->empty()) concentrationiter =
              vars.d_object_concentration[*obj].begin();
    }

    // Loop over all of the particles whose positions we know from
    // countAndCreateParticles, initialize the remaining variables
    vector<Point>::const_iterator itr;
    for(itr=vars.d_object_points[*obj].begin();
        itr!=vars.d_object_points[*obj].end(); ++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;

      if (!patch->containsPoint(*itr)) continue;
      
      particleIndex pidx = start+count;      

      // This initializes the particle values based on geom_object fields
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID, pvars);

      // Again, everything below exists for FileGeometryPiece only, where
      // a user can describe the geometry as a series of points in a file.
      // One can also describe any of the fields below in the file as well.
      // See FileGeometryPiece for usage.

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

      if (pvelocities) {
        if (!pvelocities->empty()) {
          pvars.pvelocity[pidx] = *velocityiter;
          ++velocityiter;
        }
      }

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

      if (psizes) {
        // Read psize from file or get from a smooth geometry piece
        if (!psizes->empty()) {
          pvars.psize[pidx] = *sizeiter;
          ++sizeiter;
        }
      }

      if (pareas) {
        // Read parea from file or get from a smooth geometry piece
        if (!pareas->empty()) {
          pvars.parea[pidx] = *areaiter;
          ++areaiter;
        }
      }

      if (colors) {
        if (!colors->empty()) {
          pvars.pcolor[pidx] = *coloriter;
          ++coloriter;
        }
      }

      if (concentrations) {
        if (!concentrations->empty()) {
          pvars.pConcentration[pidx] = *concentrationiter;
          ++concentrationiter;
        }
      }

      // If the particle is on the surface and if there is
      // a physical BC attached to it then mark with the 
      // physical BC pointer
      if (d_useLoadCurves) {
        if (checkForSurface(piece,*itr,dxpp)) {
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

void FileGeomPieceParticleCreator::createPoints(const Patch* patch, 
                                                GeometryObject* obj, 
                                                ObjectVars& vars)
{
}

void 
FileGeomPieceParticleCreator::initializeParticle(const Patch* patch,
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

  // The size matrix is used for storing particle domain sizes (Rvectors for
  // CPDI and CPTI) normalized by the grid spacing
  Matrix3 size(1./((double) ppc.x()),0.,0.,
               0.,1./((double) ppc.y()),0.,
               0.,0.,1./((double) ppc.z()));
  Vector area(dxpp.y()*dxpp.z(),dxpp.x()*dxpp.z(),dxpp.x()*dxpp.y());

  pvars.ptemperature[i] = (*obj)->getInitialData_double("temperature");
  pvars.plocalized[i]   = 0;

  // For AMR
  const Level* curLevel = patch->getLevel();
  pvars.prefined[i]     = curLevel->getIndex();

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
      pvars.pmass[i]   = matl->getInitialDensity()*pvars.pvolume[i]*vol_frac_CC;
     }
    } catch (...) {
      vol_frac_CC = 1.0;       
      pvars.pmass[i]      = matl->getInitialDensity()*pvars.pvolume[i];
    }
    pvars.pdisp[i]        = Vector(0.,0.,0.);

    } // end else coupledflow
  }   // end else MMS
  
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
  GeometryPieceP piece = (*obj)->getPiece();
  FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece.get_rep());
  if(fgp){
    pvars.psurface[i] = 1.0;
  } else {
    pvars.psurface[i] = checkForSurface2(piece,p,dxpp);
  }
  pvars.psurfgrad[i] = Vector(0.,0.,0.);

  Vector pExtForce(0,0,0);

  pvars.pexternalforce[i] = pExtForce;
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
FileGeomPieceParticleCreator::countAndCreateParticles(const Patch* patch, 
                                         GeometryObject* obj,
                                         ObjectVars& vars)
{
  GeometryPieceP piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getExtraBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;

  int numPts = 0;
  FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece.get_rep());
  SmoothGeomPiece   *sgp = dynamic_cast<SmoothGeomPiece*>(piece.get_rep());
  sgp->setCellSize(patch->dCell());
  if(fgp){
    fgp->setCpti(d_useCPTI);
    fgp->readPoints(patch->getID());
    numPts = fgp->returnPointCount();
  } else {
    numPts = sgp->createPoints();
  }
  vector<Point>*    points          = sgp->getPoints();
  vector<double>*   vols            = sgp->getVolume();
  vector<double>*   temps           = sgp->getTemperature();
  vector<double>*   colors          = sgp->getColors();
  vector<Vector>*   pforces         = sgp->getForces();
  vector<Vector>*   pfiberdirs      = sgp->getFiberDirs();
  vector<Vector>*   pvelocities     = sgp->getVelocity();
  vector<Matrix3>*  psizes          = sgp->getSize();
  vector<double>*   concentrations  = sgp->getConcentration();
  vector<Vector>*   pareas          = sgp->getArea();

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
      warn << "\n ERROR:MPM:FileGeomPieceParticleCreator:SmoothGeometry Piece: the point ["
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
        if (!pareas->empty() && d_doScalarDiffusion) {
          Vector psz = pareas->at(ii); 
          vars.d_object_area[obj].push_back(psz);
        }
        if (!colors->empty()) {
          double color = colors->at(ii); 
          vars.d_object_colors[obj].push_back(color);
        }
        if (!concentrations->empty()) {
          double concentration = concentrations->at(ii); 
          vars.d_object_concentration[obj].push_back(concentration);
        }
      }
    }  // patch contains cell
  }
  
  return (particleIndex) vars.d_object_points[obj].size();
}
