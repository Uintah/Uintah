#include <Packages/Uintah/Core/Grid/GeomPiece/SphereShellPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;
using namespace SCIRun;


SphereShellPiece::SphereShellPiece(ProblemSpecP& ps)
{
  setName("sphere_shell");
  ps->require("origin",d_origin);
  ps->require("radius",d_radius);
  ps->require("thickness",d_h);
  ps->require("num_lat",d_numLat);
  ps->require("num_long",d_numLong);
  
  if ( d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere radius <= 0.0"));
  if ( d_h <= 0.0)
    SCI_THROW(ProblemSetupException("Input File Error: Sphere thcknss <= 0."));
}

SphereShellPiece::~SphereShellPiece()
{
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
  Box b = patch->getBox();

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
				  ParticleVariable<Vector>& psiz,
				  particleIndex start)
{
  Box b = patch->getBox();

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
        psiz[pidx] = Vector(.5,.5,.5);
        count++;
      }
      phi += dphi;
    } 
    theta += dtheta;
  }
  
  return count;
}
