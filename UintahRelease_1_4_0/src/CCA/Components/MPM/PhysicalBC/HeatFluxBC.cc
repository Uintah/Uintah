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


#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/Box.h>
#include <iostream>

using namespace Uintah;
using namespace std;

// Store the geometry object and the load curve
HeatFluxBC::HeatFluxBC(ProblemSpecP& ps,const GridP& grid)
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
    d_surface = scinew BoxGeometryPiece(child);
    //Box box = d_surface->getBoundingBox();
    d_surfaceType = "box";
  } else if (go_type == "sphere") {
    d_surface = scinew SphereGeometryPiece(child);
    d_surfaceType = "sphere";
  } else if (go_type == "cylinder") {
    d_surface = scinew CylinderGeometryPiece(child);
    d_surfaceType = "cylinder";
  } else if (go_type == "tri") {
    d_surface = scinew TriGeometryPiece(child);
    d_surfaceType = "tri";
  } else {
    throw ParameterNotFound("** ERROR ** No surface specified for heatflux BC.",
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
    proc0cout << "\n Input File WARNING: <PhysicalBC : MPM : heat_flux> \n"
              << " The geometry Object ["<<d_surface->getType() << "] exceeds the dimensions of the computational domain.\n"
              << " \n Please change the parameters so it doesn't. \n\n"
              << " There is a flaw in the surface area calculation for the geometry object,\n"
              << " it does not take into account that the object exceeds the domain\n";
    proc0cout <<"__________________________________________________________\n";
  }
}

// Destroy the heatflux BCs
HeatFluxBC::~HeatFluxBC()
{
  delete d_surface;
  delete d_loadCurve;
}

void HeatFluxBC::outputProblemSpec(ProblemSpecP& ps)
{

  ProblemSpecP hf_ps = ps->appendChild("heat_flux");
  ProblemSpecP geom_ps = hf_ps->appendChild("geom_object");
  d_surface->outputProblemSpec(geom_ps);
  hf_ps->appendElement("numberOfParticlesOnLoadSurface",d_numMaterialPoints);
  d_loadCurve->outputProblemSpec(hf_ps);

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
    GeometryPiece* volume = scinew BoxGeometryPiece(box.lower()-dxpp, 
                                                 box.upper()+dxpp);
    if (volume->inside(p)) flag = true;
    delete volume;

  } else if (d_surfaceType == "cylinder") {
    // Create a cylindrical annulus with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length()/2.;
    CylinderGeometryPiece* cgp = dynamic_cast<CylinderGeometryPiece*>(d_surface);
    GeometryPiece* outer = scinew CylinderGeometryPiece(cgp->top(), 
                                                     cgp->bottom(), 
                                                     cgp->radius()+tol);
    GeometryPiece* inner = scinew CylinderGeometryPiece(cgp->top(), 
                                                     cgp->bottom(), 
                                                     cgp->radius()-tol);
    GeometryPiece* volume = scinew DifferenceGeometryPiece(outer, inner);
    if (volume->inside(p)) flag = true;
    delete volume;

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

  } else if (d_surfaceType == "tri") {
    // Create a spherical shell with radius-|dxpp|, radius+|dxpp|
    double tol = dxpp.length();
    TriGeometryPiece* tgp = dynamic_cast<TriGeometryPiece*>(d_surface);
    TriGeometryPiece* outer = scinew TriGeometryPiece(*tgp);
    outer->scale(1.+tol);
    
    TriGeometryPiece* inner = scinew TriGeometryPiece(*tgp);             
    inner->scale(1.-tol);

    GeometryPiece* volume = scinew DifferenceGeometryPiece(outer, inner);
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
#if 0
    cout << "theta = " << theta << " theta_n = " << theta_n << endl;
    cout << "flux = " << fluxPerParticle  << " flux_variation = " 
         << flux_variation <<  endl;
#endif

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
