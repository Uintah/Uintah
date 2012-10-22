/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/GeometryPiece/SphereShellPiece.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Malloc/Allocator.h>

#include <cmath>
#ifndef M_PI
#  define M_PI           3.14159265358979323846  /* pi */
#endif

using namespace Uintah;
using namespace SCIRun;

const string SphereShellPiece::TYPE_NAME = "sphere_shell";

SphereShellPiece::SphereShellPiece(ProblemSpecP& ps)
{
  name_ = "sphere_shell";
  ps->require("origin",d_origin);
  ps->require("radius",d_radius);
  ps->require("thickness",d_h);
  ps->require("num_lat",d_numLat);
  ps->require("num_long",d_numLong);
  
  if ( d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius <= 0.0", __FILE__, __LINE__));
  if ( d_h <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere thcknss <= 0.", __FILE__, __LINE__));
}

SphereShellPiece::~SphereShellPiece()
{
}


void SphereShellPiece::outputHelper(ProblemSpecP& sphere_ps) const
{
  sphere_ps->appendElement("origin",d_origin);
  sphere_ps->appendElement("radius",d_radius);
  sphere_ps->appendElement("thickness",d_h);
  sphere_ps->appendElement("num_lat",d_numLat);
  sphere_ps->appendElement("num_long",d_numLong);
}

GeometryPieceP
SphereShellPiece::clone() const
{
  return scinew SphereShellPiece(*this);
}

bool 
SphereShellPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else 
    return true;
}

Box 
SphereShellPiece::getBoundingBox() const
{
  Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
           d_origin.z()-d_radius);
  Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
           d_origin.z()+d_radius);
  return Box(lo,hi);
}

int 
SphereShellPiece::returnParticleCount(const Patch* patch)
{
  Box b = patch->getExtraBox();

  double dtheta = M_PI/((double) d_numLat);
  double theta = dtheta/2.;
  // Figure out the average area of each particle based on the total number
  // of particles available and the size of the sphere
  double dA     = (4.*M_PI*d_radius*d_radius)/
    (((double) d_numLat)*((double) d_numLong));

  double x,y,z;
  
  int count = 0;

  for(int i_theta=0;i_theta<d_numLat;i_theta++){
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
      x = d_origin.x() + d_radius*sin(theta)*cos(phi);
      y = d_origin.y() + d_radius*sin(theta)*sin(phi);
      z = d_origin.z() + d_radius*cos(theta);
      Point p(x,y,z);
      if(b.contains(p)){
        count++;
      }
      phi += dphi;
    }
    theta += dtheta;
  }

  return count;
}

int 
SphereShellPiece::createParticles(const Patch* patch,
                                  ParticleVariable<Point>&  pos,
                                  ParticleVariable<double>& vol,
                                  ParticleVariable<double>& pThickTop,
                                  ParticleVariable<double>& pThickBot,
                                  ParticleVariable<Vector>& pNormal,
                                  ParticleVariable<Matrix3>& psiz,
                                  particleIndex start)
{
  Box b = patch->getExtraBox();

  double dtheta =     M_PI/((double) d_numLat);
  double theta = dtheta/2.;
  // Figure out the average area of each particle based on the total number
  // of particles available and the size of the sphere
  double dA     = (4.*M_PI*d_radius*d_radius)/(double) (d_numLat*d_numLong);

  double x,y,z;
  
  int count = 0;
  
  for(int i_theta=0;i_theta<d_numLat;i_theta++){
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
      x = d_origin.x() + d_radius*sin(theta)*cos(phi);
      y = d_origin.y() + d_radius*sin(theta)*sin(phi);
      z = d_origin.z() + d_radius*cos(theta);
      Point p(x,y,z);
      if(b.contains(p)){
        particleIndex pidx = start+count;
        pos[pidx] = p;
        vol[pidx] = d_radius*d_radius*sin(theta)*dtheta*dphi*d_h;
        pThickTop[pidx] = 0.5*d_h;
        pThickBot[pidx] = 0.5*d_h;
        pNormal[pidx]  = Vector(sin(theta)*cos(phi),sin(theta)*sin(phi),
                                  cos(theta));
        psiz[pidx] = Matrix3(.5,0.,0.,
                             0.,.5,0.,
                             0.,0.,.5);
        count++;
      }
      phi += dphi;
    } 
    theta += dtheta;
  }
  
  return count;
}
