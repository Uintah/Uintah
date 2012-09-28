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


#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/MPMFlags.h>
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
PressureBC::PressureBC(ProblemSpecP& ps, const GridP& grid, const MPMFlags* flags)
{
  // First read the geometry information
  // d_surface is the geometry object containing the surface to be loaded.
  // The sign of the pressure load is +ve if applied in the direction
  // of the outward normal and -ve if applied in the direction of the
  // inward normal
  // **WARNING** Currently allows only for box, cylinder or sphere.
  if (flags->d_useCBDI){
    ps->require("outward_normal",d_outwardNormal);
  }
  d_dxpp = Vector(1.,1.,1.);  // Only needed for axisymmetric end, see below
  ProblemSpecP adult = ps->findBlock("geom_object");
  ProblemSpecP child = adult->findBlock();
  std::string go_type = child->getNodeName();
  //std::cerr << "PressureBC::go_type = " << go_type << endl;
  if (go_type == "box") {
    d_surface = scinew BoxGeometryPiece(child);
    //Box box = d_surface->getBoundingBox();
    d_surfaceType = "box";
  } else if (go_type == "sphere") {
    d_surface = scinew SphereGeometryPiece(child);
    d_surfaceType = "sphere";
  } else if (go_type == "cylinder") {
    d_surface = scinew CylinderGeometryPiece(child);
    d_surfaceType = "cylinder";
    CylinderGeometryPiece* cgp =dynamic_cast<CylinderGeometryPiece*>(d_surface);
    d_cylinder_end=cgp->cylinder_end();
    d_axisymmetric_end=cgp->axisymmetric_end();
    d_axisymmetric_side=cgp->axisymmetric_side();
    if(d_axisymmetric_end){
      ps->require("res",d_res);
      Vector dx = grid->getLevel(0)->dCell();
      d_dxpp =  Vector(dx.x()/((double) d_res.x()),
                       dx.y()/((double) d_res.y()),
                       dx.z()/((double) d_res.z()));
    }
  } else {
    throw ParameterNotFound("** ERROR ** No surface specified for pressure BC.",
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
     if(!d_axisymmetric_end && !d_axisymmetric_side){
      proc0cout <<"_________________________________________________________\n";
      proc0cout << "\n Input File WARNING: <PhysicalBC : MPM : Pressure> \n"
                << " The geometry Object ["<<d_surface->getType() << "] exceeds the dimensions of the computational domain.\n"
                << " \n Please change the parameters so it doesn't. \n\n"
                << " There is a flaw in the surface area calculation for the geometry object,\n"
                << " it does not take into account that the object exceeds the domain\n";
      proc0cout <<"_________________________________________________________\n";
    }
  }
}

// Destroy the pressure BCs
PressureBC::~PressureBC()
{
  delete d_surface;
  delete d_loadCurve;
}

void PressureBC::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP press_ps = ps->appendChild("pressure");
  ProblemSpecP geom_ps = press_ps->appendChild("geom_object");
  d_surface->outputProblemSpec(geom_ps);
  press_ps->appendElement("numberOfParticlesOnLoadSurface",d_numMaterialPoints);
  d_loadCurve->outputProblemSpec(press_ps);
  press_ps->appendElement("res",d_res);
}

// Get the type of this object for BC application
std::string 
PressureBC::getType() const
{
  return "Pressure";
}

// Locate and flag the material points to which this pressure BC is
// to be applied. Assumes that the "checkForSurface" function in ParticleCreator.cc
// has been used to identify this material point as being on the surface of the body.
// WARNING : For this logic to work, the surface object should be a 
// box (zero volume), cylinder, sphere geometry piece that touches
// contains the surface on which the pressure is to be applied.
bool
PressureBC::flagMaterialPoint(const Point& p, 
                              const Vector& dxpp)
{
  bool flag = false;
  if (d_surfaceType == "box") {
    // Create box that is min-dxpp, max+dxpp;
    Box box = d_surface->getBoundingBox();
    GeometryPiece* volume = scinew BoxGeometryPiece(box.lower()-dxpp, 
                                                 box.upper()+dxpp);

    if (volume->inside(p)) flag = true;
    delete volume;

  } else if (d_surfaceType == "cylinder") {
    double tol = 0.9*dxpp.minComponent();
    CylinderGeometryPiece* cgp =dynamic_cast<CylinderGeometryPiece*>(d_surface);

    if(!d_cylinder_end && !d_axisymmetric_end){  // Not a cylinder end
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

    }else if(d_cylinder_end || d_axisymmetric_end){
      Vector add_ends = tol*(cgp->top()-cgp->bottom())
                           /(cgp->top()-cgp->bottom()).length();

      GeometryPiece* end = scinew CylinderGeometryPiece(cgp->top()+add_ends, 
                                                        cgp->bottom()-add_ends,
                                                        cgp->radius());
      if (end->inside(p)){
         flag = true;
      }
      delete end;
    }
  } else if (d_surfaceType == "sphere") {
    // Create a spherical shell with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length();
    SphereGeometryPiece* sgp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    GeometryPiece* outer = scinew SphereGeometryPiece(sgp->origin(), 
                                                   sgp->radius()+tol);
    GeometryPiece* inner = scinew SphereGeometryPiece(sgp->origin(), 
                                                   sgp->radius()-tol);
    GeometryPiece* volume = scinew DifferenceGeometryPiece(outer, inner);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else {
    throw ParameterNotFound("ERROR: Unknown surface specified for pressure BC",
                            __FILE__, __LINE__);
  }
  
  return flag;
}

// Calculate the area of the surface on which the pressure BC
// is applied
double
PressureBC::getSurfaceArea() const
{
  double area = 0.0;
  if (d_surfaceType == "box") {
    BoxGeometryPiece* gp = dynamic_cast<BoxGeometryPiece*>(d_surface);
    area = gp->volume()/gp->smallestSide();
  } else if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    if(!d_cylinder_end && !d_axisymmetric_end){  // Not a cylinder end
      area = gp->surfaceArea();
      if(d_axisymmetric_side){
        area/=(2.0*M_PI);
      }
    }
    else if(d_cylinder_end){
      area = gp->surfaceAreaEndCaps()/2.0;
    }
    else if(d_axisymmetric_end){
      area = (gp->radius()*gp->radius())/2.0; // area of a 1 radian wedge
    }
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    area = gp->surfaceArea();
  } else {
    throw ParameterNotFound("ERROR: Unknown surface specified for pressure BC",
                            __FILE__, __LINE__);
  }
  return area;
}

// Calculate the force per particle at a certain time
double 
PressureBC::forcePerParticle(double time) const
{
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the area of the surface on which the pressure BC is applied
  double area = getSurfaceArea();

  // Get the initial pressure that is applied ( t = 0.0 )
  double press = pressure(time);

  // Calculate the forec per particle
  return (press*area)/static_cast<double>(d_numMaterialPoints);
}

// Calculate the force vector to be applied to a particular
// material point location
Vector
PressureBC::getForceVector(const Point& px, double forcePerParticle,
                           const double time) const
{
  Vector force(0.0,0.0,0.0);
  if (d_surfaceType == "box") {
    BoxGeometryPiece* gp = dynamic_cast<BoxGeometryPiece*>(d_surface);
    Vector normal(0.0, 0.0, 0.0);
    normal[gp->thicknessDirection()] = 1.0;
    force = normal*forcePerParticle;
  } else if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    Vector normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
    if(d_cylinder_end || d_axisymmetric_end){
      normal = (gp->top()-gp->bottom())
              /(gp->top()-gp->bottom()).length();
      if(!d_axisymmetric_end){
        force = normal*forcePerParticle;
      }else{  // It IS on an axisymmetric end
        double pArea = px.x()*d_dxpp.x()*1.0; /*(theta = 1 radian)*/
        double press = pressure(time);
        double fpP = pArea*press;
        force = normal*fpP;
      }
    }
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    Vector normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
  } else {
    throw ParameterNotFound("ERROR: Unknown surface specified for pressure BC",
                            __FILE__, __LINE__);
  }
  return force;
}

// Calculate the force vector to be applied to a particular
// material point location
Vector
PressureBC::getForceVectorCBDI(const Point& px, const Matrix3& psize,
                              const Matrix3& pDeformationMeasure,
                              double forcePerParticle,const double time,
                              Point& pExternalForceCorner1,
                              Point& pExternalForceCorner2,
                              Point& pExternalForceCorner3,
                              Point& pExternalForceCorner4,
                              const Vector& dxCell) const
{
  Vector force(0.0,0.0,0.0);
  Vector normal(0.0, 0.0, 0.0);
  if (d_surfaceType == "box") {
    BoxGeometryPiece* gp = dynamic_cast<BoxGeometryPiece*>(d_surface);
    normal[gp->thicknessDirection()] = 1.0;
    force = normal*forcePerParticle;
  } else if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
    if(d_cylinder_end || d_axisymmetric_end){
      normal = (gp->top()-gp->bottom())
              /(gp->top()-gp->bottom()).length();
      if(!d_axisymmetric_end){
        force = normal*forcePerParticle;
      }else{  // It IS on an axisymmetric end
        double pArea = px.x()*d_dxpp.x()*1.0; /*(theta = 1 radian)*/
        double press = pressure(time);
        double fpP = pArea*press;
        force = normal*fpP;
      }
    }
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
  } else {
    throw ParameterNotFound("ERROR: Unknown surface specified for pressure BC",
                            __FILE__, __LINE__);
  }
  // 25% of total particle force goes to each corner
  force = force*0.25;
  // modify the sign of force if outward normal is not correctly defined
  if (!d_outwardNormal) {
    force = force*(-1.0);
  }
  // determine four boundary-corners of the particle
  int i1=0,i2=0;
  Matrix3 dsize=pDeformationMeasure*psize;
  Point px1;
  for (int i = 0; i < 3; ++i) {
   Vector dummy=Vector(dsize(0,i)*dxCell[0],dsize(1,i)*dxCell[1],
                                            dsize(2,i)*dxCell[2])/2.0;
   if (abs(Dot(normal,dummy)/(normal.length()*dummy.length())-1.0)<0.1) {
    px1=Point(px.x()+dummy[0],px.y()+dummy[1],px.z()+dummy[2]);
    i1=(i+1)%3;
    i2=(i+2)%3;
   } else if (abs(Dot(normal,dummy)/(normal.length()*dummy.length())+1.0)<0.1) {
    Point px1(px.x()-dummy[0],px.y()-dummy[1],px.z()-dummy[2]);
    i1=(i+1)%3;
    i2=(i+2)%3;
   }
  }
  // px1 is the position of the center of the boundary particle face that is on the physical boundary.
  pExternalForceCorner1=Point(px1.x()-dsize(0,i1)*dxCell[0]/2.0-dsize(0,i2)*dxCell[0]/2.0,
                              px1.y()-dsize(1,i1)*dxCell[1]/2.0-dsize(1,i2)*dxCell[1]/2.0,
                              px1.z()-dsize(2,i1)*dxCell[2]/2.0-dsize(2,i2)*dxCell[2]/2.0);
  pExternalForceCorner2=Point(px1.x()+dsize(0,i1)*dxCell[0]/2.0-dsize(0,i2)*dxCell[0]/2.0,
                              px1.y()+dsize(1,i1)*dxCell[1]/2.0-dsize(1,i2)*dxCell[1]/2.0,
                              px1.z()+dsize(2,i1)*dxCell[2]/2.0-dsize(2,i2)*dxCell[2]/2.0);
  pExternalForceCorner3=Point(px1.x()-dsize(0,i1)*dxCell[0]/2.0+dsize(0,i2)*dxCell[0]/2.0,
                              px1.y()-dsize(1,i1)*dxCell[1]/2.0+dsize(1,i2)*dxCell[1]/2.0,
                              px1.z()-dsize(2,i1)*dxCell[2]/2.0+dsize(2,i2)*dxCell[2]/2.0);
  pExternalForceCorner4=Point(px1.x()+dsize(0,i1)*dxCell[0]/2.0+dsize(0,i2)*dxCell[0]/2.0,
                              px1.y()+dsize(1,i1)*dxCell[1]/2.0+dsize(1,i2)*dxCell[1]/2.0,
                              px1.z()+dsize(2,i1)*dxCell[2]/2.0+dsize(2,i2)*dxCell[2]/2.0);
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
// A method to print out the pressure bcs
ostream& operator<<(ostream& out, const PressureBC& bc) 
{
   out << "Begin MPM Pressure BC # = " << bc.loadCurveID() << endl;
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
         << " pressure = " << lc->getLoad(ii) << endl;
   }
   out << "End MPM Pressure BC # = " << bc.loadCurveID() << endl;
   return out;
}

} // end namespace Uintah
