#include <Packages/Uintah/Core/Grid/SphereMembraneGeometryPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;
using namespace SCIRun;


SphereMembraneGeometryPiece::SphereMembraneGeometryPiece(ProblemSpecP& ps)
{

  Point orig;
  double rad,h;

  ps->require("origin",orig);
  ps->require("radius",rad);
  ps->require("thickness",h);
  
  if ( rad <= 0.0)
   throw ProblemSetupException("Input File Error: Sphere radius must be > 0.0");
  if ( h <= 0.0)
   throw ProblemSetupException("Input File Error: Sphere thcknss must be > 0.");
  
  d_origin = orig;
  d_radius = rad;
  d_h = h;
}

SphereMembraneGeometryPiece::~SphereMembraneGeometryPiece()
{
}

bool SphereMembraneGeometryPiece::inside(const Point& p) const
{
  Vector diff = p - d_origin;

  if (diff.length() > d_radius)
    return false;
  else 
    return true;
  
}

Box SphereMembraneGeometryPiece::getBoundingBox() const
{
    Point lo(d_origin.x()-d_radius,d_origin.y()-d_radius,
	   d_origin.z()-d_radius);

    Point hi(d_origin.x()+d_radius,d_origin.y()+d_radius,
	   d_origin.z()+d_radius);

    return Box(lo,hi);

}

int SphereMembraneGeometryPiece::returnParticleCount(const Patch* patch)
{
  Box b = patch->getBox();

  int d_numLat  = 20;
  int d_numLong = 40;

  double PI     = 3.14159265359;
  double dtheta =     PI/((double) d_numLat);
  double dphi   = 2.0*PI/((double) d_numLong);

  double theta = dtheta/2.;
  double phi   = dphi/2.;
  double x,y,z;
  
  int count = 0;

  for(int i_theta=0;i_theta<d_numLat;i_theta++){
    phi   = dphi/2.;
    for(int i_phi=0;i_phi<d_numLong;i_phi++){
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
                                                  particleIndex start)
{
  Box b = patch->getBox();

  int d_numLat  = 20;
  int d_numLong = 40;

  double PI     = 3.14159265359;
  double dtheta =     PI/((double) d_numLat);
  double dphi   = 2.0*PI/((double) d_numLong);

  double theta = dtheta/2.;
  double phi   = dphi/2.;
  double x,y,z;

  int count = 0;

  for(int i_theta=0;i_theta<d_numLat;i_theta++){
    phi   = dphi/2.;
    for(int i_phi=0;i_phi<d_numLong;i_phi++){
      x = d_origin.x() + d_radius*sin(theta)*cos(phi);
      y = d_origin.y() + d_radius*sin(theta)*sin(phi);
      z = d_origin.z() + d_radius*cos(theta);
      Point p(x,y,z);
      if(b.contains(p)){
        pos[start+count] = p;
        vol[start+count] = d_radius*d_radius*sin(theta)*dtheta*dphi*d_h;
        count++;
      }
      phi += dphi;
    }
    theta += dtheta;
  }

  return count;

}
