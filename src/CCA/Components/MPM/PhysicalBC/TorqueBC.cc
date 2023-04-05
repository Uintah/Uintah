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

#include <CCA/Components/MPM/PhysicalBC/TorqueBC.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/Matrix3.h>
#include <iostream>

using namespace Uintah;
using namespace std;

// Store the geometry object and the load curve
TorqueBC::TorqueBC(ProblemSpecP& ps, const GridP& grid,
                       const MPMFlags* flags)
{
  // First read the geometry information
  // d_surface is the geometry object containing the surface to be loaded.
  // The sign of the torque load is +ve if applied in the direction
  // of the outward normal and -ve if applied in the direction of the
  // inward normal
  // **WARNING** Currently allows only for box, cylinder or sphere.
  if (flags->d_useCBDI){
    ps->require("outward_normal",d_outwardNormal);
  }
  ProblemSpecP adult = ps->findBlock("geom_object");
  ProblemSpecP child = adult->findBlock();
  std::string go_type = child->getNodeName();
  //std::cerr << "TorqueBC::go_type = " << go_type << endl;
  if (go_type == "cylinder") {
    d_surface = scinew CylinderGeometryPiece(child);
    d_surfaceType = "cylinder";
    CylinderGeometryPiece* cgp =dynamic_cast<CylinderGeometryPiece*>(d_surface);
    d_axis = cgp->top() - cgp->bottom();
  } else {
    throw ParameterNotFound("ERROR: Only cylinders work with torque BC",
                            __FILE__, __LINE__);
  }

  d_numMaterialPoints = 0;  // this value is read in on a restart
  ps->get("numberOfParticlesOnLoadSurface",d_numMaterialPoints);

  // Read and save the load curve information
  d_loadCurve = scinew LoadCurve<double>(ps);

  //__________________________________
  //   Bulletproofing
  // user shouldn't specify a geometry object that is bigger than the domain
  Box boundingBox = d_surface->getBoundingBox();
  BBox compDomain;
  grid->getSpatialRange(compDomain);
  
  Point BB_min = boundingBox.lower();
  Point CD_min = compDomain.min();
  Point BB_max = boundingBox.upper();
  Point CD_max = compDomain.max(); 
  
  if( ( BB_min.x() < CD_min.x() ) ||
      ( BB_min.y() < CD_min.y() ) || 
      ( BB_min.z() < CD_min.z() ) ||
      ( BB_max.x() > CD_max.x() ) ||
      ( BB_max.y() > CD_max.y() ) ||
      ( BB_max.z() > CD_max.z() ) ){
//     if(!d_axisymmetric_end && !d_axisymmetric_side){
      proc0cout <<"_________________________________________________________\n";
      proc0cout << "\n Input File WARNING: <PhysicalBC : MPM : Torque> \n"
                << " The geometry Object ["<<d_surface->getType() << "] exceeds the dimensions of the computational domain.\n"
                << " \n Please change the parameters so it doesn't. \n\n"
                << " There is a flaw in the surface area calculation for the geometry object,\n"
                << " it does not take into account that the object exceeds the domain\n";
      proc0cout <<"_________________________________________________________\n";
//    }
  }
}

// Destroy the torque BCs
TorqueBC::~TorqueBC()
{
  delete d_surface;
  delete d_loadCurve;
}

void TorqueBC::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP torque_ps = ps->appendChild("torque");
  ProblemSpecP geom_ps = torque_ps->appendChild("geom_object");
  d_surface->outputProblemSpec(geom_ps);
  torque_ps->appendElement("numberOfParticlesOnLoadSurface",
                                                           d_numMaterialPoints);
  d_loadCurve->outputProblemSpec(torque_ps);
  torque_ps->appendElement("outward_normal",d_outwardNormal);
}

// Get the type of this object for BC application
std::string 
TorqueBC::getType() const
{
  return "Torque";
}

// Locate and flag the material points to which this torque BC is
// to be applied. Assumes that the "checkForSurface" function in ParticleCreator.cc
// has been used to identify this material point as being on the surface of the body.
// WARNING : For this logic to work, the surface object should be a 
// box (zero volume), cylinder, sphere geometry piece that touches
// contains the surface on which the torque is to be applied.
bool
TorqueBC::flagMaterialPoint(const Point& p, 
                              const Vector& dxpp)
{
  bool flag = false;
  if (d_surfaceType == "cylinder") {
    double tol = 0.9*dxpp.minComponent();
    CylinderGeometryPiece* cgp =dynamic_cast<CylinderGeometryPiece*>(d_surface);

//    if(!d_cylinder_end && !d_axisymmetric_end){  // Not a cylinder end
    // Create a cylindrical annulus with radius-|dxpp|, radius+|dxpp|
    GeometryPiece* outer = scinew CylinderGeometryPiece(cgp->top(), 
                                                        cgp->bottom(), 
                                                        cgp->radius()+tol);

    GeometryPiece* inner = scinew CylinderGeometryPiece(cgp->top(), 
                                                        cgp->bottom(), 
                                                        cgp->radius()-tol);

    GeometryPiece* volume = scinew DifferenceGeometryPiece(outer, inner);

    if (volume->inside(p)){
      flag = true;
    }
    delete volume;

//    }
  } else {
    throw ParameterNotFound("ERROR: Only cylinders work with torque BC",
                            __FILE__, __LINE__);
  }
  
  return flag;
}

// Calculate the area of the surface on which the torque BC
// is applied
double
TorqueBC::getSurfaceArea() const
{
  double area = 0.0;
  if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
//    if(!d_cylinder_end && !d_axisymmetric_end){  // Not a cylinder end
    area = gp->surfaceArea();
//    }
  } else {
    throw ParameterNotFound("ERROR: Only cylinders work with torque BC",
                            __FILE__, __LINE__);
  }
  return area;
}

// Calculate the torque per particle at a certain time
double 
TorqueBC::torquePerParticle(double time) const
{
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the area of the surface on which the torque BC is applied
//  double area = getSurfaceArea();

  // Get the initial torque that is applied ( t = 0.0 )
  double torque = Torque(time);

  // Calculate the forec per particle
  return (torque)/static_cast<double>(d_numMaterialPoints);
}

// Calculate the force vector to be applied to a particular
// material point location
Vector
TorqueBC::getForceVector(const Point& px, double torquePerParticle,
                           const double time) const
{
  Vector force(0.0,0.0,0.0);
  if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);

    // dir is the magnitude of the radius and the direction of the force vector
    Vector dir = Cross(px - gp->bottom(), d_axis)/d_axis.length();
    double dirlength = dir.length();
    force = torquePerParticle*(dir/(dirlength*dirlength));
//    cout << "px = " << px << endl;
//    cout << "dir = " << dir << endl;
//    cout << "forceDir = " << forceDir << endl;
//    cout << "force = " << force << endl;
//    cout << "forceMag = " << force.length() << endl << endl;
  } else {
    throw ParameterNotFound("ERROR: Only cylinders work with torque BC",
                            __FILE__, __LINE__);
  }
  return force;
}

// Calculate the force vector to be applied to a particular
// material point location
Vector
TorqueBC::getForceVectorCBDI(const Point& px, const Matrix3& psize,
                              const Matrix3& pDeformationMeasure,
                              double forcePerParticle,const double time,
                              Point& pExternalForceCorner1,
                              Point& pExternalForceCorner2,
                              Point& pExternalForceCorner3,
                              Point& pExternalForceCorner4,
                              const Vector& dx) const
{
  Vector force(0.0,0.0,0.0);
  Vector normal(0.0, 0.0, 0.0);
  if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
  } else {
    throw ParameterNotFound("ERROR: Only cylinder sides work with torque BC",
                            __FILE__, __LINE__);
  }
  // 25% of total particle force goes to each corner
  force = force*0.25;
  // modify the sign of force if outward normal is not correctly defined
  if (!d_outwardNormal) {
    normal = normal*(-1.0);
  }
  // determine four boundary-corners of the particle
  // px1 is the position of the center of the boundary particle face
  // that is on the physical boundary.
  // The direction to px1 is determined by taking the dot product of the
  // normal vector with each of the r-vectors.  the one with a unity dot product
  // is the winner.  Then just go out to that surface.
  // TODO:  Bailing out of the loop as soon as px1 is found.  
  int i1=0,i2=0;
  Matrix3 dsize=pDeformationMeasure*psize;
  Point px1;
  for (int i = 0; i < 3; ++i) {
   Vector dummy=Vector(dsize(0,i)*dx[0],dsize(1,i)*dx[1],dsize(2,i)*dx[2])*0.5;
   if (abs(Dot(normal,dummy)/(normal.length()*dummy.length())-1.0)<0.1) {
    px1=Point(px.x()+dummy[0],px.y()+dummy[1],px.z()+dummy[2]);
    i1=(i+1)%3;
    i2=(i+2)%3;
   } else if (abs(Dot(normal,dummy)/(normal.length()*dummy.length())+1.0)<0.1) {
    px1 = Point(px.x()-dummy[0],px.y()-dummy[1],px.z()-dummy[2]);
    i1=(i+1)%3;
    i2=(i+2)%3;
   }
  }
  pExternalForceCorner1=Point(px1.x()-.5*(dsize(0,i1)*dx[0]-dsize(0,i2)*dx[0]),
                              px1.y()-.5*(dsize(1,i1)*dx[1]-dsize(1,i2)*dx[1]),
                              px1.z()-.5*(dsize(2,i1)*dx[2]-dsize(2,i2)*dx[2]));
  pExternalForceCorner2=Point(px1.x()+.5*(dsize(0,i1)*dx[0]-dsize(0,i2)*dx[0]),
                              px1.y()+.5*(dsize(1,i1)*dx[1]-dsize(1,i2)*dx[1]),
                              px1.z()+.5*(dsize(2,i1)*dx[2]-dsize(2,i2)*dx[2]));
  pExternalForceCorner3=Point(px1.x()-.5*(dsize(0,i1)*dx[0]+dsize(0,i2)*dx[0]),
                              px1.y()-.5*(dsize(1,i1)*dx[1]+dsize(1,i2)*dx[1]),
                              px1.z()-.5*(dsize(2,i1)*dx[2]+dsize(2,i2)*dx[2]));
  pExternalForceCorner4=Point(px1.x()+.5*(dsize(0,i1)*dx[0]+dsize(0,i2)*dx[0]),
                              px1.y()+.5*(dsize(1,i1)*dx[1]+dsize(1,i2)*dx[1]),
                              px1.z()+.5*(dsize(2,i1)*dx[2]+dsize(2,i2)*dx[2]));

  // Recalculate the force based on area changes (current vs. initial)
  Vector iniVec1(psize(0,i1),psize(1,i1),psize(2,i1));
  Vector iniVec2(psize(0,i2),psize(1,i2),psize(2,i2));
  Vector curVec1(dsize(0,i1),dsize(1,i1),dsize(2,i1));
  Vector curVec2(dsize(0,i2),dsize(1,i2),dsize(2,i2));
  Vector iniA = Cross(iniVec1,iniVec2);
  Vector curA = Cross(curVec1,curVec2);
  double iniArea=iniA.length();
  double curArea=curA.length();
  force=force*(curArea/iniArea);
  return force;
}

namespace Uintah {
// A method to print out the torque bcs
ostream& operator<<(ostream& out, const TorqueBC& bc) 
{
   out << "Begin MPM Torque BC # = " << bc.loadCurveID() << endl;
   std::string surfType = bc.getSurfaceType();
   out << "    Surface of application = " << surfType << endl;
   if (surfType == "box") {
      Box box = (bc.getSurface())->getBoundingBox();
      out << "        " << box << endl;
   } else if (surfType == "cylinder") {
      CylinderGeometryPiece* cgp = 
         dynamic_cast<CylinderGeometryPiece*>(bc.getSurface());
      out << "        " << "radius = " << cgp->radius() 
                        << " top = " << cgp->top() 
                        << " bottom = " << cgp->bottom() << endl;
   } else if (surfType == "sphere") {
      SphereGeometryPiece* sgp = 
         dynamic_cast<SphereGeometryPiece*>(bc.getSurface());
      out << "        " << "radius = " << sgp->radius() 
                        << " origin = " << sgp->origin() << endl;
   }
   out << "    Time vs. Load = " << endl;
   LoadCurve<double>* lc = bc.getLoadCurve();
   int numPts = lc->numberOfPointsOnLoadCurve();
   for (int ii = 0; ii < numPts; ++ii) {
     out << "        time = " << lc->getTime(ii) 
         << " torque = " << lc->getLoad(ii) << endl;
   }
   out << "End MPM Torque BC # = " << bc.loadCurveID() << endl;
   return out;
}

} // end namespace Uintah
