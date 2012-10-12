/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/GeometryPiece/GUVSphereShellPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <cmath>

#ifndef M_PI
#  define M_PI           3.14159265358979323846  /* pi */
#endif

using namespace Uintah;
using namespace SCIRun;
using std::cout;

const string GUVSphereShellPiece::TYPE_NAME = "guv_sphere_shell";

GUVSphereShellPiece::GUVSphereShellPiece(ProblemSpecP& ps)
{
  name_ = "Unnamed GUV_shell";
  ps->require("origin",d_origin);
  ps->require("radius",d_radius);
  if ( d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius <= 0.0", __FILE__, __LINE__));
  ps->require("thickness_lipid",d_h_lipid);
  ps->require("thickness_cholesterol",d_h_cholesterol);
  if ( d_h_lipid <= 0.0 || d_h_cholesterol <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere thcknss <= 0.", __FILE__, __LINE__));
  for (ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()) {
    std::string zone_type = child->getNodeName();
    if (zone_type == "zone") {
      double theta, phi, radius;
      child->require("zone_center_theta",theta);
      child->require("zone_center_phi",phi);
      child->require("zone_radius",radius);
      d_theta_zone.push_back(theta*M_PI/180.0);
      d_phi_zone.push_back(phi*M_PI/180.0);
      d_radius_zone.push_back(radius);
    }
  }
}

GUVSphereShellPiece::~GUVSphereShellPiece()
{
}

void
GUVSphereShellPiece::outputHelper( ProblemSpecP & ps ) const
{
  ProblemSpecP shell_ps = ps->appendChild("shell");
  ProblemSpecP guv_ps = shell_ps->appendChild("GUV_sphere");

  guv_ps->appendElement("origin",d_origin);
  guv_ps->appendElement("radius",d_radius);
  guv_ps->appendElement("thickness_lipid",d_h_lipid);
  guv_ps->appendElement("thickness_cholesterol",d_h_cholesterol);

  for (unsigned int i = 0; i <= d_theta_zone.size(); i++) {
    ProblemSpecP zone_ps = guv_ps->appendChild("zone");
    zone_ps->appendElement("zone_center_theta",d_theta_zone[i]*180./M_PI );
    zone_ps->appendElement("zone_center_phi",d_phi_zone[i]*180./M_PI);
    zone_ps->appendElement("zone_center_radius",d_radius_zone[i]);
  }
}


GeometryPieceP
GUVSphereShellPiece::clone() const
{
  return scinew GUVSphereShellPiece(*this);
}

bool 
GUVSphereShellPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else 
    return true;
}

Box 
GUVSphereShellPiece::getBoundingBox() const
{
  Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
           d_origin.z()-d_radius);
  Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
           d_origin.z()+d_radius);
  return Box(lo,hi);
}

int 
GUVSphereShellPiece::createPoints()
{
  int numPts = d_pos.size();
  if (numPts > 0) return numPts;

  // Compute the number of latitudes
  double numLat = (int) (2.0*M_PI*d_radius/d_dx);
  double dtheta = M_PI/((double) numLat);
  double theta = dtheta/2.0;

  // Figure out the average area of each particle based on the total number
  // of particles available and the size of the sphere
  double dA = (4.*M_PI*d_radius*d_radius)/(double) (numLat*numLat);

  double x,y,z;
  int count = 0;
  for(int i_theta=0;i_theta<numLat;i_theta++){

    // compute a different dphi for each theta to keep the particle size
    // approximately constant
    double dphi   = dA/(d_radius*d_radius*sin(theta)*dtheta);
    double numLong = ((int)(2*M_PI/dphi));

    // adjust the number of particle circumferentially so that the number
    // of particles around is always divisible by four.  This makes using
    // symmetry BCs possible.
    numLong = numLong + (4-((int) numLong)%4);
    dphi = (2*M_PI)/numLong;
    double phi   = dphi/2.;
    for(int i_phi=0;i_phi<numLong;i_phi++){
      ++count;
      x = d_origin.x() + d_radius*sin(theta)*cos(phi);
      y = d_origin.y() + d_radius*sin(theta)*sin(phi);
      z = d_origin.z() + d_radius*cos(theta);
      Point p(x,y,z);

      // Store the quantities that are the same for both the lipid and
      // the cholesterol
      d_pos.push_back(p);
      d_norm.push_back(Vector(sin(theta)*cos(phi),sin(theta)*sin(phi),
                              cos(theta)));

      if(insideZone(p)){
        d_vol.push_back(d_radius*d_radius*sin(theta)*dtheta*dphi*d_h_cholesterol);
        d_thick.push_back(d_h_cholesterol);
        d_type.push_back(1);
      } else {
        d_vol.push_back(d_radius*d_radius*sin(theta)*dtheta*dphi*d_h_lipid);
        d_thick.push_back(d_h_lipid);
        d_type.push_back(0);
      }
      phi += dphi;
    } 
    theta += dtheta;
  }
  
  return count;
}

//////////
/*! Find if a point is inside a cholesterol zone */
bool 
GUVSphereShellPiece::insideZone(const Point& p) const
{
  // Find the cholesterol zones
  int numZones = d_radius_zone.size();
  for (int zone = 0; zone < numZones; ++zone) {
    double theta = d_theta_zone[zone];
    double phi = d_phi_zone[zone];
    double rad = d_radius_zone[zone];
    double x = d_origin.x() + d_radius*sin(theta)*cos(phi);
    double y = d_origin.y() + d_radius*sin(theta)*sin(phi);
    double z = d_origin.z() + d_radius*cos(theta);
    Vector cen(x,y,z);
    Vector normal(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    Vector axis = normal*d_radius;
    Vector bot = cen - axis;
    Vector tobot = p.asVector() - bot;
    double height2 = d_radius*d_radius;
    double h = Dot(tobot, axis);
    if(h < 0.0 || h > height2) continue; // Above or below the cylinder
    double area = Cross(axis, tobot).length2();
    double d = area/height2;
    if( d > rad*rad) continue;
    return true;
  }
  return false;
}

int 
GUVSphereShellPiece::createParticles(const Patch* ,
                                     ParticleVariable<Point>&  ,
                                     ParticleVariable<double>& ,
                                     ParticleVariable<double>& ,
                                     ParticleVariable<double>& ,
                                     ParticleVariable<Vector>& ,
                                     ParticleVariable<Matrix3>& ,
                                     particleIndex )
{
  cout << "**ERROR**GUVSphereShellPiece:: No create particles implemented.";
  return 0;
}

int 
GUVSphereShellPiece::returnParticleCount(const Patch*)
{
  cout << "**ERROR**GUVSphereShellPiece:: No per patch particle count.";
  return 0;
}

