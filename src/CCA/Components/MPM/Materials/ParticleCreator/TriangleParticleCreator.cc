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

#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
//#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

#include <iostream>

/*  This code is specialized for filling triangulated surfaces based on
a method developed by Duan Zhang, et al., at Sandia.  It is faster and
less memory intensive than the original method which uses the "inside" operators
in the TriGeometryPiece code.

This code is a bit tough to follow.  Here's the basic order of operations.

First, MPM::actuallyInitialize calls MPMMaterial::createParticles, which in
turn calls ParticleCreator::createParticles for the appropriate ParticleCreator
MPMMaterial calls the ParticleCreatorFactory::create

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

createPoints stacks the triangles in a single direction, so that the first check
is to see if a point in the other two directions is even in a cell that has
triangles.  If it does, then all of the candidate locations in that single
direction are checked to see how many triangles they pass through.  Even means
the point is outside the object, odd means it is inside.  Because of
(apparently) round-off error, it is necessary to do this in each of the three
directions independently, and if a point is determined to be inside in at least
two of those, it is kept.

Now that we know how many particles we have for this material on this patch,
we are ready to allocateVariables, which calls allocateAndPut for all of the
variables needed in SerialMPM or AMRMPM.  At this point, storage for the
particles has been created, but the arrays allocated are still empty.

Now back in createParticles, the next step is to loop over all of the 
GeometryObjects. Either way, loop over all of the particles in
object points and initialize the remaining particle data.  This is done for
by calling initializeParticle.  

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
}

TriangleParticleCreator::~TriangleParticleCreator()
{
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

  pvars.ptempPrevious[i]  = pvars.ptemperature[i];
  if(d_flags->d_useLogisticRegression ||
     d_useLoadCurves){
    GeometryPieceP piece = (*obj)->getPiece();
    pvars.psurface[i] = checkForSurface(piece,p,dxpp);
  } else {
    pvars.psurface[i] = 0.;
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
