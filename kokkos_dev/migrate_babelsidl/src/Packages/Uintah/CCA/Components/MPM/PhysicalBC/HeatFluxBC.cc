#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/GeometryPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/TriGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SphereGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/DifferenceGeometryPiece.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Store the geometry object and the load curve
HeatFluxBC::HeatFluxBC(ProblemSpecP& ps)
{
  // First read the geometry information
  // d_surface is the geometry object containing the surface to be loaded.
  // The sign of the heatflux load is +ve if applied in the direction
  // of the outward normal and -ve if applied in the direction of the
  // inward normal
  // **WARNING** Currently allows only for box, cylinder or sphere.
  ProblemSpecP child = (ps->findBlock("geom_object"))->findBlock();
  std::string go_type = child->getNodeName();
  //std::cerr << "HeatFluxBC::go_type = " << go_type << endl;
  if (go_type == "box") {
    d_surface = new BoxGeometryPiece(child);
    //Box box = d_surface->getBoundingBox();
    d_surfaceType = "box";
  } else if (go_type == "sphere") {
    d_surface = new SphereGeometryPiece(child);
    d_surfaceType = "sphere";
  } else if (go_type == "cylinder") {
    d_surface = new CylinderGeometryPiece(child);
    d_surfaceType = "cylinder";
  } else if (go_type == "tri") {
    d_surface = new TriGeometryPiece(child);
    d_surfaceType = "tri";
  } else {
    throw ParameterNotFound("** ERROR ** No surface specified for heatflux BC.",
                            __FILE__, __LINE__);
  }
  d_numMaterialPoints = 0;

  // Read and save the load curve information
  d_loadCurve = new LoadCurve<double>(ps); 
}

// Destroy the heatflux BCs
HeatFluxBC::~HeatFluxBC()
{
  delete d_surface;
  delete d_loadCurve;
}

// Get the type of this object for BC application
std::string 
HeatFluxBC::getType() const
{
  return "HeatFlux";
}

// Locate and flag the material points to which this heatflux BC is
// to be applied. Assumes that the "checkForSurface" function in ParticleCreator.cc
// has been used to identify this material point as being on the surface of the body.
// WARNING : For this logic to work, the surface object should be a 
// box (zero volume), cylinder, sphere geometry piece that touches
// contains the surface on which the heatflux is to be applied.
bool
HeatFluxBC::flagMaterialPoint(const Point& p, 
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
    double tol = dxpp.length()/2.;
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

  } else if (d_surfaceType == "tri") {
    // Create a spherical shell with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length();
    TriGeometryPiece* tgp = dynamic_cast<TriGeometryPiece*>(d_surface);
    TriGeometryPiece* outer = new TriGeometryPiece(*tgp);
    outer->scale(1.+tol);
    
    TriGeometryPiece* inner = new TriGeometryPiece(*tgp);             
    inner->scale(1.-tol);

    GeometryPiece* volume = new DifferenceGeometryPiece(outer, inner);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for heatflux BC",
                            __FILE__, __LINE__);
  }
  
  return flag;
}

// Calculate the area of the surface on which the heatflux BC
// is applied
double
HeatFluxBC::getSurfaceArea() const
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
  } else if (d_surfaceType == "tri") {
    TriGeometryPiece* gp = dynamic_cast<TriGeometryPiece*>(d_surface);
    area = gp->surfaceArea();
  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for heatflux BC",
                            __FILE__, __LINE__);
  }
  return area;
}

// Calculate the force per particle at a certain time
double 
HeatFluxBC::fluxPerParticle(double time) const
{
  //cout << "d_numMaterialPoints = " << d_numMaterialPoints << endl;
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the area of the surface on which the heatflux BC is applied
  double area = getSurfaceArea();
  //double area = 1;

  // Get the initial heatflux that is applied ( t = 0.0 )
  double heatflx = heatflux(time);
  //  cout << "heatflx = " << heatflx << endl;

  // Calculate the heatflux per particle
  return (heatflx*area)/static_cast<double>(d_numMaterialPoints);
}

// Calculate the flux vector to be applied to a particular
// material point location
double
HeatFluxBC::getFlux(const Point& px, double fluxPerParticle) const
{
  double flux(0.0);
  if (d_surfaceType == "box") {
    BoxGeometryPiece* gp = dynamic_cast<BoxGeometryPiece*>(d_surface);
    Vector normal(0.0, 0.0, 0.0);
    normal[gp->thicknessDirection()] = 1.0;
    flux = fluxPerParticle;
  } else if (d_surfaceType == "cylinder") {
    CylinderGeometryPiece* gp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    Vector normal = gp->radialDirection(px);
    double theta = atan(px.y()/px.x());
    double theta_n = atan(normal.y()/normal.x());
    double max_min = .3;  // span for which the flux varies over the surface
    double flux_variation_mag = fluxPerParticle*max_min/2.;
    double offset = fluxPerParticle - flux_variation_mag;
    double flux_variation = flux_variation_mag*cos(theta) + offset;
    cout << "theta = " << theta << " theta_n = " << theta_n << endl;
    cout << "flux = " << fluxPerParticle  << " flux_variation = " 
         << flux_variation <<  endl;

    flux = fluxPerParticle;
  } else if (d_surfaceType == "sphere") {
    SphereGeometryPiece* gp = dynamic_cast<SphereGeometryPiece*>(d_surface);
    Vector normal = gp->radialDirection(px);
    flux = fluxPerParticle;
  } else {
    throw ParameterNotFound("** ERROR ** Unknown surface specified for heatflux BC",
                            __FILE__, __LINE__);
  }
  return flux;
}

namespace Uintah {
// A method to print out the heatflux bcs
ostream& operator<<(ostream& out, const HeatFluxBC& bc) 
{
   out << "Begin MPM HeatFlux BC # = " << bc.loadCurveID() << endl;
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
       //         << " heatflux = " << lc->getLoad(ii) << endl;
         << " heatflux = " << bc.heatflux(ii) << endl;
   }
   out << "End MPM HeatFlux BC # = " << bc.loadCurveID() << endl;
   return out;
}

} // end namespace Uintah
