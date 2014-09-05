/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

using namespace Uintah;
using namespace std;

// Store the geometry object and the load curve
PressureBC::PressureBC(ProblemSpecP& ps, const GridP& grid)
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
    proc0cout <<"__________________________________________________________\n";
    proc0cout << "\n Input File WARNING: <PhysicalBC : MPM : Pressure> \n"
              << " The geometry Object ["<<d_surface->getType() << "] exceeds the dimensions of the computational domain.\n"
              << " \n Please change the parameters so it doesn't. \n\n"
              << " There is a flaw in the surface area calculation for the geometry object,\n"
              << " it does not take into account that the object exceeds the domain\n";
    proc0cout <<"__________________________________________________________\n";
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
  d_dxpp=dxpp;
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
