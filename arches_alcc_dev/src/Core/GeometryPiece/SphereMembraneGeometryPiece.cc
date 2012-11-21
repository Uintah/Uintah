/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/GeometryPiece/SphereMembraneGeometryPiece.h>

#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Box.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>

#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

#include <cmath>

using namespace Uintah;
using namespace SCIRun;

const string SphereMembraneGeometryPiece::TYPE_NAME = "sphere_membrane";

SphereMembraneGeometryPiece::SphereMembraneGeometryPiece(ProblemSpecP& ps)
{
  name_ = "Unnamed Sphere_membrane";

  ps->require("origin",d_origin);
  ps->require("radius",d_radius);
  ps->require("thickness",d_h);
  ps->require("num_lat",d_numLat);
  ps->require("num_long",d_numLong);
  
  if ( d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius must be > 0.0", __FILE__, __LINE__));
  if ( d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere thcknss must be > 0.", __FILE__, __LINE__));
}

SphereMembraneGeometryPiece::~SphereMembraneGeometryPiece()
{
}

void
SphereMembraneGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("origin",d_origin);
  ps->appendElement("radius",d_radius);
  ps->appendElement("thickness",d_h);
  ps->appendElement("num_lat",d_numLat);
  ps->appendElement("num_long",d_numLong);
}

GeometryPieceP
SphereMembraneGeometryPiece::clone() const
{
  return scinew SphereMembraneGeometryPiece(*this);
}

bool
SphereMembraneGeometryPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;

  if (diff.length() > d_radius)
    return false;
  else 
    return true;
}

Box
SphereMembraneGeometryPiece::getBoundingBox() const
{
    Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
             d_origin.z()-d_radius);

    Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
             d_origin.z()+d_radius);

    return Box(lo,hi);

}

int
SphereMembraneGeometryPiece::returnParticleCount(const Patch* patch)
{
  Box b = patch->getExtraBox();

  double PI     = 3.14159265359;
  double dtheta =     PI/((double) d_numLat);
  double theta = dtheta/2.;
  // Figure out the average area of each particle based on the total number
  // of particles available and the size of the sphere
  double dA     = (4.*PI*d_radius*d_radius)/
                  (((double) d_numLat)*((double) d_numLong));

  double x,y,z;
  
  int count = 0;

  for(int i_theta=0;i_theta<d_numLat;i_theta++){
    // compute a different dphi for each theta to keep the particle size
    // approximately constant
    double dphi   = dA/(d_radius*d_radius*sin(theta)*dtheta);
    double numLong = ((int)(2*PI/dphi));
    // adjust the number of particle circumferentially so that the number
    // of particles around is always divisible by four.  This makes using
    // symmetry BCs possible.
    numLong = numLong + (4-((int) numLong)%4);
    dphi = (2*PI)/numLong;
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

int SphereMembraneGeometryPiece::createParticles(const Patch* patch,
                                                 ParticleVariable<Point>&  pos,
                                                 ParticleVariable<double>& vol,
                                                 ParticleVariable<Vector>& pt1,
                                                 ParticleVariable<Vector>& pt2,
                                                 ParticleVariable<Vector>& pn,
                                                 ParticleVariable<Matrix3>& psiz,
                                                 particleIndex start)
{
  Box b = patch->getExtraBox();

  double PI     = 3.14159265359;
  double dtheta =     PI/((double) d_numLat);
  double theta = dtheta/2.;
  // Figure out the average area of each particle based on the total number
  // of particles available and the size of the sphere
  double dA     = (4.*PI*d_radius*d_radius)/
                  (((double) d_numLat)*((double) d_numLong));

  double x,y,z;
  
  int count = 0;
  
  for(int i_theta=0;i_theta<d_numLat;i_theta++){
    // compute a different dphi for each theta to keep the particle size
    // approximately constant
    double dphi   = dA/(d_radius*d_radius*sin(theta)*dtheta);
    double numLong = ((int)(2*PI/dphi));
    // adjust the number of particle circumferentially so that the number
    // of particles around is always divisible by four.  This makes using
    // symmetry BCs possible.
    numLong = numLong + (4-((int) numLong)%4);
    dphi = (2*PI)/numLong;
    double phi   = dphi/2.;
    for(int i_phi=0;i_phi<numLong;i_phi++){
      x = d_origin.x() + d_radius*sin(theta)*cos(phi);
      y = d_origin.y() + d_radius*sin(theta)*sin(phi);
      z = d_origin.z() + d_radius*cos(theta);
      Point p(x,y,z);
      if(b.contains(p)){
        pos[start+count] = p;
        vol[start+count] = d_radius*d_radius*sin(theta)*dtheta*dphi*d_h;
        pt1[start+count] = Vector(cos(theta)*cos(phi),cos(theta)*sin(phi),
                                                                -sin(theta));
        pt2[start+count] = Vector(-sin(phi),cos(phi),0);
        pn[start+count]  = Vector(sin(theta)*cos(phi),sin(theta)*sin(phi),
                                                                 cos(theta));
        psiz[start+count] = Matrix3(.5,0.,0.,
                             0.,.5,0.,
                             0.,0.,.5);

//        psiz[start+count]= Vector(fabs(-d_radius*sin(theta)*dphi*sin(phi)
//                                +       d_radius*dtheta*cos(theta)*cos(phi)
//                                +       dx.x()*sin(theta)*cos(phi))/dx.x(),
//
//                                  fabs(d_radius*sin(theta)*dphi*cos(phi)
//                                +      d_radius*dtheta*cos(theta)*sin(phi)
//                                +      dx.x()*sin(theta)*sin(phi))/dx.x(),
//                                  fabs(-d_radius*dtheta*sin(theta)
//                                +       dx.x()*cos(theta))/dx.x());

        count++;
      }
      phi += dphi;
    } 
    theta += dtheta;
  }
  
  return count;
}
