#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SphereGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/DifferenceGeometryPiece.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Store the geometry object and the load curve
PressureBC::PressureBC(ProblemSpecP& ps)
{
  // First read the geometry information
  // d_surface is the geometry object containing the surface to be loaded.
  // The sign of the pressure load is +ve if applied in the direction
  // of the outward normal and -ve if applied in the direction of the
  // inward normal
  // **WARNING** Currently allows only for box, cylinder or sphere.
  ProblemSpecP child = (ps->findBlock("geom_object"))->findBlock();
  std::string go_type = child->getNodeName();
  //std::cerr << "PressureBC::go_type = " << go_type << endl;
  if (go_type == "box") {
    d_surface = new BoxGeometryPiece(child);
    Box box = d_surface->getBoundingBox();
    d_surfaceType = "box";
  } else if (go_type == "sphere") {
    d_surface = new SphereGeometryPiece(child);
    d_surfaceType = "sphere";
  } else if (go_type == "cylinder") {
    d_surface = new CylinderGeometryPiece(child);
    d_surfaceType = "cylinder";
  } else {
    throw ParameterNotFound("** ERROR ** No surface specified for pressure BC.");
  }
  d_numMaterialPoints = 0;

  // Read and save the load curve information
  d_loadCurve = new LoadCurve<double>(ps); 
}

// Destroy the pressure BCs
PressureBC::~PressureBC()
{
  delete d_surface;
  delete d_loadCurve;
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
                              const Vector& dxpp) const
{
  bool flag = false;
  if (d_surfaceType == "box") {
    // Create box that is min-dxpp, max+dxpp;
    Box box = d_surface->getBoundingBox();
    GeometryPiece* volume = new BoxGeometryPiece(box.lower()-dxpp, 
                                                 box.upper()+dxpp);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else if (d_surfaceType == "cylinder") {
    // Create a cylindrical annulus with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length();
    CylinderGeometryPiece* cgp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    GeometryPiece* outer = new CylinderGeometryPiece(cgp->top(), 
						     cgp->bottom(), 
						     cgp->radius()+tol);
    GeometryPiece* inner = new CylinderGeometryPiece(cgp->top(), 
						     cgp->bottom(), 
						     cgp->radius()-tol);
    GeometryPiece* volume = new DifferenceGeometryPiece(outer, inner);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else if (d_surfaceType == "sphere") {
    // Create a spherical shell with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length();
    SphereGeometryPiece* sgp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    GeometryPiece* outer = new SphereGeometryPiece(sgp->origin(), 
						   sgp->radius()+tol);
    GeometryPiece* inner = new SphereGeometryPiece(sgp->origin(), 
						   sgp->radius()-tol);
    GeometryPiece* volume = new DifferenceGeometryPiece(outer, inner);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for pressure BC");
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
    area = gp->surfaceArea();
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    area = gp->surfaceArea();
  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for pressure BC");
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
PressureBC::getForceVector(const Point& px, double forcePerParticle) const
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
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    Vector normal = gp->radialDirection(px);
    force = normal*forcePerParticle;
  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for pressure BC");
  }
  return force;
}

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

