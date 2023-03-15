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

#include <CCA/Components/MPM/Materials/ParticleCreator/TriangleParticleCreator.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/MMS/MMS.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
//#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

#include <iostream>

/*  This code is a bit tough to follow.  Here's the basic order of operations.

First, MPM::actuallyInitialize calls MPMMaterial::createParticles, which in
turn calls ParticleCreator::createParticles for the appropriate ParticleCreator
(MPMMaterial calls the ParticleCreatorFactory::create, which is kind of stupid
since every material will use the same type ParticleCreator. Whatever..)

Next,  createParticles, below, first loops over all of the geom_objects and
calls countAndCreateParticles.  countAndCreateParticles returns the number of
particles on a given patch associated with each geom_object and accumulates
that into a variable called num_particles.  countAndCreateParticles gets
the number of particles by either querying the functions for smooth geometry 
piece types, or by calling createPoints, also below.  When createPoints is
called, as each particle is determined to be inside of the object, it is pushed
back into the object_points entry of the ObjectVars struct.  ObjectVars
consists of several maps which are indexed on the GeometryObject and a vector
containing whatever data that entry is responsible for carrying.  A map is used
because even after particles are created, their initial data is still tied
back to the GeometryObject.  These might include velocity, temperature, color,
etc.

createPoints, for the non-smooth geometry, essentially visits each cell,
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
GeometryObjects.  If the GeometryObject is a SmoothGeometryPiece, those
type of objects MAY have their own methods for populating the data within the
if(sgp) conditional.  Either way, loop over all of the particles in
object points and initialize the remaining particle data.  This is done for
non-Smooth/File pieces by calling initializeParticle.  For the Smooth/File
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

TriangleParticleCreator::TriangleParticleCreator(MPMMaterial* matl, 
                                 MPMFlags* flags)
                              :  ParticleCreator(matl,flags)
{

  d_lb = scinew MPMLabel();
  d_useLoadCurves = flags->d_useLoadCurves;
  d_with_color = flags->d_with_color;
  d_artificial_viscosity = flags->d_artificial_viscosity;
  d_computeScaleFactor = flags->d_computeScaleFactor;

  d_flags = flags;

  registerPermanentParticleState(matl);
}

TriangleParticleCreator::~TriangleParticleCreator()
{
  delete d_lb;
}

particleIndex 
TriangleParticleCreator::createParticles(MPMMaterial* matl,
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

    // Loop over all of the particles whose positions we know from
    // countAndCreateParticles, initialize the remaining variables
    vector<Point>::const_iterator itr;
    for(itr=vars.d_object_points[*obj].begin();
        itr!=vars.d_object_points[*obj].end(); ++itr){
      IntVector cell_idx;
      if (!patch->findCell(*itr,cell_idx)) continue;

      if (!patch->containsPoint(*itr)) continue;
      
      particleIndex pidx = start+count;      

      // This initializes the particle values for objects
      initializeParticle(patch,obj,matl,*itr,cell_idx,pidx,cellNAPID, pvars);

      // If the particle is on the surface and if there is
      // a physical BC attached to it then mark with the 
      // physical BC pointer
      if (d_useLoadCurves) {
        // DO WE NEED THIS, OR USE PSURFACE ALREADY COMPUTED?
        if (checkForSurface(piece,*itr,dxpp)) {
          Vector areacomps;
          pvars.pLoadCurveID[pidx] = getLoadCurveID(*itr, dxpp, areacomps, dwi);
        } else {
          pvars.pLoadCurveID[pidx] = IntVector(0,0,0);
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
IntVector TriangleParticleCreator::getLoadCurveID(const Point& pp, const Vector& dxpp,
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
    else if (bcs_type == "HeatFlux") {      
      HeatFluxBC* hfbc = 
        dynamic_cast<HeatFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      if (hfbc->flagMaterialPoint(pp, dxpp)) {
         ret(k) = hfbc->loadCurveID(); 
         k++;
      }
    }
  }
  return ret;
}

// Print MPM physical boundary condition information
void TriangleParticleCreator::printPhysicalBCs()
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
  }
}

ParticleSubset* 
TriangleParticleCreator::allocateVariables(particleIndex numParticles, 
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
  new_dw->allocateAndPut(pvars.psurface,      d_lb->pSurfLabel,         subset);
  new_dw->allocateAndPut(pvars.pmodalID,      d_lb->pModalIDLabel,      subset);
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

  if(d_artificial_viscosity){
     new_dw->allocateAndPut(pvars.p_q,        d_lb->p_qLabel,           subset);
  }
  return subset;
}

void TriangleParticleCreator::createPoints(const Patch* patch, 
                                           GeometryObject* obj, 
                                           ObjectVars& vars)
{
  // In this version of create points, triangles are projected into
  // planes and points are checked to see if they lie within the projection
  // of those triangles.  If they do, then the out of plane direction is
  // checked to see if the point is inside of an object. 
  GeometryPieceP piece = obj->getPiece();
  Box b2 = patch->getExtraBox();
  IntVector ppc = obj->getInitialData_IntVector("res");

  Vector DX = patch->dCell();
  Vector dxpp = DX/ppc;
  Vector dcorner = dxpp*0.5;

  IntVector high = patch->getCellHighIndex();
  IntVector low  = patch->getCellLowIndex();

  Point anchor = patch->getLevel()->getAnchor();

  TriGeometryPiece *tgp = dynamic_cast<TriGeometryPiece*>(piece.get_rep());

  vector<IntVector> triangles = tgp->getTriangles();
  vector<Point>     points    = tgp->getPoints();

  // create vectors to store the max/min x and y for each triangle
  int numTriangles = triangles.size();
  vector<double> triXmax(numTriangles);
  vector<double> triYmax(numTriangles);
  vector<double> triZmax(numTriangles);
  vector<double> triXmin(numTriangles);
  vector<double> triYmin(numTriangles);
  vector<double> triZmin(numTriangles);

  //cout << "numTriangles = " << numTriangles << endl;
  //cout << "numPoints = " << points.size() << endl;
  // Find min/max x and y for each triangle
  double xa, xb, xc;
  double ya, yb, yc;
  double za, zb, zc;
  for(int i=0;i<numTriangles;i++){
    xa = points[triangles[i].x()].x(); 
    xb = points[triangles[i].y()].x(); 
    xc = points[triangles[i].z()].x();

    ya = points[triangles[i].x()].y(); 
    yb = points[triangles[i].y()].y(); 
    yc = points[triangles[i].z()].y();

    za = points[triangles[i].x()].z(); 
    zb = points[triangles[i].y()].z(); 
    zc = points[triangles[i].z()].z();
    triXmin[i]=min(min(xa,xb),xc);
    triXmax[i]=max(max(xa,xb),xc);
    triYmin[i]=min(min(ya,yb),yc);
    triYmax[i]=max(max(ya,yb),yc);
    triZmin[i]=min(min(za,zb),zc);
    triZmax[i]=max(max(za,zb),zc);
  }  // Loop over all triangles

  double epsilon = 1.e-8;

  set<Point> zProjSet;
  set<Point> yProjSet;
  set<Point> xProjSet;

  // First, project all triangles into the x-y plane
  for(int i=low.x(); i<high.x(); i++){
   double xcorner = ((double) i)*DX.x()+dcorner.x() + anchor.x();
   for(int ix=0; ix < ppc.x(); ix++){
    double xcand = xcorner + ((double) ix)*dxpp.x();
    for(int j=low.y(); j<high.y(); j++){
      double ycorner = ((double) j)*DX.y()+dcorner.y() + anchor.y();
      for(int iy=0; iy < ppc.y(); iy++){
        double ycand = ycorner + ((double) iy)*dxpp.y();
        vector<double> ZI;
        for(int m=0;m<numTriangles;m++){
          // First, check to see if candidate point is inside of a rectangle 
          // that contains the triangle.  If it is, do more math.
          if(triXmin[m]<=xcand && xcand<=triXmax[m]){
            if(triYmin[m]<=ycand && ycand<=triYmax[m]){
              // inside the bounding square, compute Eqs 3 and 4
              double ebx = points[triangles[m].y()].x() 
                         - points[triangles[m].x()].x();
              double eby = points[triangles[m].y()].y() 
                         - points[triangles[m].x()].y();
              double ecx = points[triangles[m].z()].x() 
                         - points[triangles[m].x()].x();
              double ecy = points[triangles[m].z()].y() 
                         - points[triangles[m].x()].y();
              double ebDOTec = ebx*ecx + eby*ecy;
              double ebDOTeb = ebx*ebx + eby*eby;
              double ecDOTeb = ebDOTec;
              double ecDOTec = ecx*ecx + ecy*ecy;
              double ebPx = ebDOTec*ebx - ebDOTeb*ecx;
              double ebPy = ebDOTec*eby - ebDOTeb*ecy;
              double ecPx = ecDOTec*ebx - ecDOTeb*ecx;
              double ecPy = ecDOTec*eby - ecDOTeb*ecy;
              double wx = xcand - points[triangles[m].x()].x();
              double wy = ycand - points[triangles[m].x()].y();
              double u = (wx*ecPx + wy*ecPy)/(ebx*ecPx + eby*ecPy);
              double v = (wx*ebPx + wy*ebPy)/(ecx*ebPx + ecy*ebPy);
              if(u>=-1.*epsilon && v>=-1.*epsilon && u+v<=1.+epsilon){
                // x-y point is within a triangle
                // Compute zi for each triangle that gets this far
                double zi = u*(points[triangles[m].y()].z() 
                             - points[triangles[m].x()].z())
                          + v*(points[triangles[m].z()].z() 
                             - points[triangles[m].x()].z()) + 
                               points[triangles[m].x()].z();
                ZI.push_back(zi);
              } // Is point inside of triangle
            }   // Is point in bounding rectangle y
          }     // Is point in bounding rectangle x
        }       // Loop over triangles       

        if(ZI.size()>0){
         // elimintate any double counts due to expanding triangles
         sort(ZI.begin(), ZI.end());
         vector<double> ZIsorted;
         ZIsorted.push_back(ZI[0]);
         for(unsigned int izi=1;izi<ZI.size();izi++){
           if(fabs(ZI[izi] - ZI[izi-1]) > 2.*epsilon){
             ZIsorted.push_back(ZI[izi]);
           }
         }

         for(int k=low.z(); k<high.z(); k++){
           double zcorner = ((double) k)*DX.z()+dcorner.z() + anchor.z();
           for(int iz=0;iz < ppc.z(); iz++){
             double zcand = zcorner + ((double) iz)*dxpp.z();
             int count = 0;
             for(unsigned int izi=0;izi<ZIsorted.size();izi++){
               if(ZIsorted[izi]>zcand){
                 count++;
               }
             }
             if(count%2==1){
               Point p(xcand, ycand, zcand);
               //vars.d_object_points[obj].push_back(p);
               zProjSet.insert(p);
             }
           }  // z
         }  // k-cell
       }  // if ZI isn't empty
      }  // y
    }  // j-cell
   }  // x
  }  // i-cell

  // Next, project all triangles into the x-z plane
  for(int i=low.x(); i<high.x(); i++){
   double xcorner = ((double) i)*DX.x()+dcorner.x() + anchor.x();
   for(int ix=0; ix < ppc.x(); ix++){
    double xcand = xcorner + ((double) ix)*dxpp.x();
    for(int k=low.z(); k<high.z(); k++){
      double zcorner = ((double) k)*DX.z()+dcorner.z() + anchor.z();
      for(int iz=0; iz < ppc.z(); iz++){
        double zcand = zcorner + ((double) iz)*dxpp.z();
        vector<double> YI;
        for(int m=0;m<numTriangles;m++){
          // First, check to see if candidate point is inside of a rectangle 
          // that contains the triangle.  If it is, do more math.
          if(triXmin[m]<=xcand && xcand<=triXmax[m]){
            if(triZmin[m]<=zcand && zcand<=triZmax[m]){
              // inside the bounding square, compute Eqs 3 and 4
              double ebx = points[triangles[m].y()].x() 
                         - points[triangles[m].x()].x();
              double ebz = points[triangles[m].y()].z() 
                         - points[triangles[m].x()].z();
              double ecx = points[triangles[m].z()].x() 
                         - points[triangles[m].x()].x();
              double ecz = points[triangles[m].z()].z() 
                         - points[triangles[m].x()].z();
              double ebDOTec = ebx*ecx + ebz*ecz;
              double ebDOTeb = ebx*ebx + ebz*ebz;
              double ecDOTeb = ebDOTec;
              double ecDOTec = ecx*ecx + ecz*ecz;
              double ebPx = ebDOTec*ebx - ebDOTeb*ecx;
              double ebPz = ebDOTec*ebz - ebDOTeb*ecz;
              double ecPx = ecDOTec*ebx - ecDOTeb*ecx;
              double ecPz = ecDOTec*ebz - ecDOTeb*ecz;
              double wx = xcand - points[triangles[m].x()].x();
              double wz = zcand - points[triangles[m].x()].z();
              double u = (wx*ecPx + wz*ecPz)/(ebx*ecPx + ebz*ecPz);
              double v = (wx*ebPx + wz*ebPz)/(ecx*ebPx + ecz*ebPz);
              if(u>=-1.*epsilon && v>=-1.*epsilon && u+v<=1.+epsilon){
                // x-y point is within a triangle
                // Compute yi for each triangle that gets this far
                double yi = u*(points[triangles[m].y()].y() 
                             - points[triangles[m].x()].y())
                          + v*(points[triangles[m].z()].y() 
                             - points[triangles[m].x()].y()) + 
                               points[triangles[m].x()].y();
                YI.push_back(yi);
              } // Is point inside of triangle
            }   // Is point in bounding rectangle y
          }     // Is point in bounding rectangle x
        }       // Loop over triangles       

        if(YI.size()>0){
         // elimintate any double counts due to expanding triangles
         sort(YI.begin(), YI.end());
         vector<double> YIsorted;
         YIsorted.push_back(YI[0]);
         for(unsigned int iyi=1;iyi<YI.size();iyi++){
           if(fabs(YI[iyi] - YI[iyi-1]) > 2.*epsilon){
             YIsorted.push_back(YI[iyi]);
           }
         }

         for(int j=low.y(); j<high.y(); j++){
           double ycorner = ((double) j)*DX.y()+dcorner.y() + anchor.y();
           for(int iy=0;iy < ppc.y(); iy++){
             double ycand = ycorner + ((double) iy)*dxpp.y();
             int count = 0;
             for(unsigned int iyi=0;iyi<YIsorted.size();iyi++){
               if(YIsorted[iyi]>ycand){
                 count++;
               }
             }
             if(count%2==1){
               Point p(xcand, ycand, zcand);
               //vars.d_object_points[obj].push_back(p);
               yProjSet.insert(p);
             }
           }  // y
         }  // j-cell
       }  // if YI isn't empty
      }  // z
    }  // k-cell
   }  // x
  }  // i-cell

  // Finally, project all triangles into the y-z plane
  for(int j=low.y(); j<high.y(); j++){
   double ycorner = ((double) j)*DX.y()+dcorner.y() + anchor.y();
   for(int iy=0; iy < ppc.y(); iy++){
    double ycand = ycorner + ((double) iy)*dxpp.y();
    for(int k=low.z(); k<high.z(); k++){
      double zcorner = ((double) k)*DX.z()+dcorner.z() + anchor.z();
      for(int iz=0; iz < ppc.z(); iz++){
        double zcand = zcorner + ((double) iz)*dxpp.z();
        vector<double> XI;
        for(int m=0;m<numTriangles;m++){
          // First, check to see if candidate point is inside of a rectangle 
          // that contains the triangle.  If it is, do more math.
          if(triYmin[m]<=ycand && ycand<=triYmax[m]){
            if(triZmin[m]<=zcand && zcand<=triZmax[m]){
              // inside the bounding square, compute Eqs 3 and 4
              double eby = points[triangles[m].y()].y() 
                         - points[triangles[m].x()].y();
              double ebz = points[triangles[m].y()].z() 
                         - points[triangles[m].x()].z();
              double ecy = points[triangles[m].z()].y() 
                         - points[triangles[m].x()].y();
              double ecz = points[triangles[m].z()].z() 
                         - points[triangles[m].x()].z();
              double ebDOTec = eby*ecy + ebz*ecz;
              double ebDOTeb = eby*eby + ebz*ebz;
              double ecDOTeb = ebDOTec;
              double ecDOTec = ecy*ecy + ecz*ecz;
              double ebPy = ebDOTec*eby - ebDOTeb*ecy;
              double ebPz = ebDOTec*ebz - ebDOTeb*ecz;
              double ecPy = ecDOTec*eby - ecDOTeb*ecy;
              double ecPz = ecDOTec*ebz - ecDOTeb*ecz;
              double wy = ycand - points[triangles[m].x()].y();
              double wz = zcand - points[triangles[m].x()].z();
              double u = (wy*ecPy + wz*ecPz)/(eby*ecPy + ebz*ecPz);
              double v = (wy*ebPy + wz*ebPz)/(ecy*ebPy + ecz*ebPz);
              if(u>=-1.*epsilon && v>=-1.*epsilon && u+v<=1.+epsilon){
                // z-y point is within a triangle
                // Compute xi for each triangle that gets this far
                double xi = u*(points[triangles[m].y()].x() 
                             - points[triangles[m].x()].x())
                          + v*(points[triangles[m].z()].x() 
                             - points[triangles[m].x()].x()) + 
                               points[triangles[m].x()].x();
                XI.push_back(xi);
              } // Is point inside of triangle
            }   // Is point in bounding rectangle y
          }     // Is point in bounding rectangle x
        }       // Loop over triangles       

        if(XI.size()>0){
         // elimintate any double counts due to expanding triangles
         sort(XI.begin(), XI.end());
         vector<double> XIsorted;
         XIsorted.push_back(XI[0]);
         for(unsigned int ixi=1;ixi<XI.size();ixi++){
           if(fabs(XI[ixi] - XI[ixi-1]) > 2.*epsilon){
             XIsorted.push_back(XI[ixi]);
           }
         }

         for(int i=low.x(); i<high.x(); i++){
           double xcorner = ((double) i)*DX.x()+dcorner.x() + anchor.x();
           for(int ix=0;ix < ppc.x(); ix++){
             double xcand = xcorner + ((double) ix)*dxpp.x();
             int count = 0;
             for(unsigned int ixi=0;ixi<XIsorted.size();ixi++){
               if(XIsorted[ixi]>xcand){
                 count++;
               }
             }
             if(count%2==1){
               Point p(xcand, ycand, zcand);
               //vars.d_object_points[obj].push_back(p);
               xProjSet.insert(p);
             }
           }  // y
         }  // j-cell
       }  // if XI isn't empty
      }  // z
    }  // k-cell
   }  // x
  }  // i-cell

  // First create intersections between each of the pairs of projected
  // directions.  First YZ, then XZ, then XY.  Then insert all of the points
  // from those intersection into union set that will contain all particles
  // that considered to be inside for at least two of the projected directions
  set<Point> intersectZY;
  set_intersection(zProjSet.begin(), zProjSet.end(), 
                   yProjSet.begin(), yProjSet.end(),
                   std::inserter(intersectZY, intersectZY.begin()));

  set<Point> intersectZX;
  set_intersection(zProjSet.begin(), zProjSet.end(), 
                   xProjSet.begin(), xProjSet.end(),
                   std::inserter(intersectZX, intersectZX.begin()));

  set<Point> intersectYX;
  set_intersection(yProjSet.begin(), yProjSet.end(), 
                   xProjSet.begin(), xProjSet.end(),
                   std::inserter(intersectYX, intersectYX.begin()));

  // Set containing all of the points that will be created as particles
  set<Point> unionXY_XZ_YZ;

  for (set<Point>::iterator it1 = intersectZY.begin();
                            it1!= intersectZY.end();  it1++){
    unionXY_XZ_YZ.insert(*it1);
  }

  for (set<Point>::iterator it1 = intersectZX.begin();
                            it1!= intersectZX.end();  it1++){
    unionXY_XZ_YZ.insert(*it1);
  }

  for (set<Point>::iterator it1 = intersectYX.begin();
                            it1!= intersectYX.end();  it1++){
    unionXY_XZ_YZ.insert(*it1);
  }

  // Finally, push all of the points in the union of the intersections
  // into the vector of points to become particles.
  for (set<Point>::iterator it1 = unionXY_XZ_YZ.begin();
                            it1!= unionXY_XZ_YZ.end();  it1++){
    Point point = *it1;
    vars.d_object_points[obj].push_back(point);
  }

}

void 
TriangleParticleCreator::initializeParticle(const Patch* patch,
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

  pvars.position[i] = p;
  // standard voxel volume
  pvars.pvolume[i]  = size.Determinant()*dxcc.x()*dxcc.y()*dxcc.z();
  pvars.psize[i]      = size;  // Normalized by grid spacing
  pvars.pvelocity[i]  = (*obj)->getInitialData_Vector("velocity");
  pvars.pvelGrad[i]  = Matrix3(0.0);
  pvars.pTempGrad[i] = Vector(0.0);
  pvars.pmass[i]     = matl->getInitialDensity()*pvars.pvolume[i];
  pvars.pdisp[i]     = Vector(0.,0.,0.);

  if(d_with_color){
    pvars.pcolor[i] = (*obj)->getInitialData_double("color");
  }
  if(d_artificial_viscosity){
    pvars.p_q[i] = 0.;
  }

  pvars.ptempPrevious[i]  = pvars.ptemperature[i];
  GeometryPieceP piece = (*obj)->getPiece();
  pvars.psurface[i] = 1.0; //checkForSurface2(piece,p,dxpp);
  pvars.pmodalID[i]  = matl->getModalID();
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
TriangleParticleCreator::countAndCreateParticles(const Patch* patch, 
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

vector<const VarLabel* > TriangleParticleCreator::returnParticleState()
{
  return particle_state;
}


vector<const VarLabel* > TriangleParticleCreator::returnParticleStatePreReloc()
{
  return particle_state_preReloc;
}

void TriangleParticleCreator::registerPermanentParticleState(MPMMaterial* matl)
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

  particle_state.push_back(d_lb->pVelGradLabel);
  particle_state_preReloc.push_back(d_lb->pVelGradLabel_preReloc);

  if (d_flags->d_refineParticles) {
    particle_state.push_back(d_lb->pRefinedLabel);
    particle_state_preReloc.push_back(d_lb->pRefinedLabel_preReloc);
  }

  particle_state.push_back(d_lb->pStressLabel);
  particle_state_preReloc.push_back(d_lb->pStressLabel_preReloc);

  particle_state.push_back(d_lb->pLocalizedMPMLabel);
  particle_state_preReloc.push_back(d_lb->pLocalizedMPMLabel_preReloc);

  if(d_flags->d_useLogisticRegression || 
       d_flags->d_SingleFieldMPM      ||
       d_flags->d_doingDissolution){
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

  if (d_computeScaleFactor) {
    particle_state.push_back(d_lb->pScaleFactorLabel);
    particle_state_preReloc.push_back(d_lb->pScaleFactorLabel_preReloc);
  }

  particle_state.push_back(d_lb->pModalIDLabel);
  particle_state_preReloc.push_back(d_lb->pModalIDLabel_preReloc);

  matl->getConstitutiveModel()->addParticleState(particle_state,
                                                 particle_state_preReloc);
                                                 
  matl->getDamageModel()->addParticleState( particle_state, particle_state_preReloc );
  
  matl->getErosionModel()->addParticleState( particle_state, particle_state_preReloc );
}

int
TriangleParticleCreator::checkForSurface( const GeometryPieceP piece, const Point p,
                                  const Vector dxpp )
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
TriangleParticleCreator::checkForSurface2(const GeometryPieceP piece, const Point p,
                                  const Vector dxpp )
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
    return 1.0;
  } else {
    return 0.0;
  }
}
