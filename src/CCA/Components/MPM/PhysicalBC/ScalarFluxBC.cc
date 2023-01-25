/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
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
#include <iostream>

using namespace Uintah;
using namespace std;

// Store the geometry object and the load curve
ScalarFluxBC::ScalarFluxBC(ProblemSpecP& ps, const GridP& grid,
                       const MPMFlags* flags)
{
  // First read the geometry information
  // d_surface is the geometry object containing the surface to be loaded.
#if 0
  // The sign of the scalar flux load is +ve if applied in the direction
  // of the outward normal and -ve if applied in the direction of the
  // inward normal
  // **WARNING** Currently allows only for box, cylinder or sphere.
  if (flags->d_useCBDI){
    ps->require("outward_normal",d_outwardNormal);
  }
#endif

  d_dxpp = Vector(1.,1.,1.);  // Only needed for axisymmetric end, see below
  ProblemSpecP adult = ps->findBlock("geom_object");
  ProblemSpecP child = adult->findBlock();
  std::string go_type = child->getNodeName();
  //std::cerr << "ScalarFluxBC::go_type = " << go_type << endl;
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
    throw ParameterNotFound("* ERROR *: No surface specified for ScalarFluxBC.",
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
      proc0cout << "\n Input File WARNING: <PhysicalBC : MPM : ScalarFlux> \n"
                << " The geometry Object ["<<d_surface->getType() << "] exceeds the dimensions of the computational domain.\n"
                << " \n Please change the parameters so it doesn't. \n\n"
                << " There is a flaw in the surface area calculation for the geometry object,\n"
                << " it does not take into account that the object exceeds the domain\n";
      proc0cout <<"_________________________________________________________\n";
    }
  }
}

// Destroy the ScalarFlux BCs
ScalarFluxBC::~ScalarFluxBC()
{
  delete d_surface;
  delete d_loadCurve;
}

void ScalarFluxBC::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP press_ps = ps->appendChild("scalar_flux");
  ProblemSpecP geom_ps = press_ps->appendChild("geom_object");
  d_surface->outputProblemSpec(geom_ps);
  press_ps->appendElement("numberOfParticlesOnLoadSurface",d_numMaterialPoints);
  d_loadCurve->outputProblemSpec(press_ps);
  press_ps->appendElement("outward_normal",d_outwardNormal);
  if(d_axisymmetric_end){
    press_ps->appendElement("res",d_res);
  }
}

// Get the type of this object for BC application
std::string 
ScalarFluxBC::getType() const
{
  return "ScalarFlux";
}

// Locate and flag the material points to which this ScalarFlux BC is
// to be applied. Assumes that the "checkForSurface" function in ParticleCreator.cc
// has been used to identify this material point as being on the surface of the body.
// WARNING : For this logic to work, the surface object should be a 
// box (zero volume), cylinder, sphere geometry piece that touches
// contains the surface on which the ScalarFlux is to be applied.
bool
ScalarFluxBC::flagMaterialPoint(const Point& p, 
                                const Vector& dxpp,
                                Vector& areacomps)
{

  bool flag = false;
  if (d_surfaceType == "box") {
    // Create box that is min-dxpp, max+dxpp;
    Box box = d_surface->getBoundingBox();
    GeometryPiece* volume = scinew BoxGeometryPiece(box.lower()-dxpp, 
                                                    box.upper()+dxpp);

    if (volume->inside(p)){
      flag = true;
      Vector diff = box.upper()-box.lower();
      if(diff.minComponent()==diff.x()){
        areacomps = Vector(1.0,0.0,0.0);
      } else if(diff.minComponent()==diff.y()){
        areacomps = Vector(0.0,1.0,0.0);
      } else if(diff.minComponent()==diff.z()){
        areacomps = Vector(0.0,0.0,1.0);
      }
    }
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
      double normalX = p.x() - 0.5*(cgp->bottom()+cgp->top()).x();
      double normalY = p.y() - 0.5*(cgp->bottom()+cgp->top()).y();
      double length = sqrt(normalX*normalX + normalY*normalY);
      areacomps = Vector(normalX/length,normalY/length,0.0);
      delete volume;

    }else if(d_cylinder_end || d_axisymmetric_end){
      Vector add_ends = tol*(cgp->top()-cgp->bottom())
                           /(cgp->top()-cgp->bottom()).length();

      GeometryPiece* end = scinew CylinderGeometryPiece(cgp->top()   +add_ends, 
                                                        cgp->bottom()-add_ends,
                                                        cgp->radius());
      if (end->inside(p)){
         flag = true;
      }
      areacomps = Vector(0.0,0.0,1.0); // Area normal to the axial direction
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
    areacomps = Vector(1.0,0.0,0.0); // Area normal to the radial direction
    delete volume;

  } else {
    throw ParameterNotFound("ERROR: Unknown surface specified for ScalarFluxBC",
                            __FILE__, __LINE__);
  }
  return flag;
}

#if 0
// Calculate the area of the surface on which the scalar flux BC
// is applied
double
ScalarFluxBC::getSurfaceArea() const
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
    throw ParameterNotFound("ERROR: Unknown surface specified for ScalarFluxBC",
                            __FILE__, __LINE__);
  }
  return area;
}

// Calculate the flux per particle at a certain time
double ScalarFluxBC::fluxPerParticle(double time) const
{
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the area of the surface on which the scalar flux BC is applied
  double area = getSurfaceArea();

  // Get the initial scalar flux that is applied ( t = 0.0 )
  double flux = ScalarFlux(time);

  // Calculate the forec per particle
  return (flux*area)/static_cast<double>(d_numMaterialPoints);
}
#endif

// Calculate the flux at a certain time given the area of a particular particle
double ScalarFluxBC::fluxPerParticle(double time, double area) const
{
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the initial scalar flux that is applied ( t = 0.0 )
  double flux = ScalarFlux(time);

  // Calculate dC/dt per particle -- JBH, 9/26/2017
  // Fixme TODO Make sure area is correct!
  return flux*area;
}

namespace Uintah {
// A method to print out the scalar flux bcs
ostream& operator<<(ostream& out, const ScalarFluxBC& bc) 
{
   out << "Begin MPM ScalarFlux BC # = " << bc.loadCurveID() << endl;
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
         << " scalar flux = " << lc->getLoad(ii) << endl;
   }
   out << "End MPM ScalarFlux BC # = " << bc.loadCurveID() << endl;
   return out;
}

} // end namespace Uintah
