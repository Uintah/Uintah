#include <Packages/Uintah/Core/Grid/GeomPiece/PlaneShellPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

//////////
// Constructor : Initialize stuff
PlaneShellPiece::PlaneShellPiece(ProblemSpecP& ps)
{
  setName("plane");
  ps->require("center", d_center);
  ps->require("normal", d_normal);
  ps->require("radius", d_radius);
  ps->require("thickness", d_thickness);
  ps->require("num_radius", d_numRadius);
  
  if (d_normal.length2() < 1.0)
    SCI_THROW(ProblemSetupException("PlaneShell: Incorrect normal in input file."));
  if (d_radius <= 0.0 || d_thickness <= 0.0)
    SCI_THROW(ProblemSetupException("PlaneShell: Incorrect plane dimensions."));
  if (d_numRadius < 1.0)
    SCI_THROW(ProblemSetupException("PlaneShell: Incorrect subdivision of plane."));

  cout << "Creating a plane shell" << endl;
  cout << "   Center = " << d_center << endl;
  cout << "   Normal = " << d_normal << endl;
  cout << "   Radius = " << d_radius << endl;
  cout << "   Thickness = " << d_thickness << endl;
  cout << "   Particles in radial direction = " << d_numRadius << endl;
}

//////////
// Destructor
PlaneShellPiece::~PlaneShellPiece()
{
}

//////////
// Find if a point is inside the cylinder
bool 
PlaneShellPiece::inside(const Point& p) const
{
  double halfThick = 0.5*d_thickness;
  double height2 = d_thickness*d_thickness;
  Vector axis = d_normal*d_thickness;
  Vector bot = d_center.asVector() - d_normal*halfThick;
  Vector tobot = p.asVector() - bot;
  double h = Dot(tobot, axis);
  if(h < 0.0 || h > height2)
    return false; // Above or below the cylinder

  double area = Cross(axis, tobot).length2();
  double d = area/height2;
  if( d > d_radius*d_radius)
    return false;
  return true;
}

//////////
// Find the bounding box for the cylinder
Box 
PlaneShellPiece::getBoundingBox() const
{
  double halfThick = 0.5*d_thickness;
  Vector bot = d_center.asVector() - d_normal*halfThick;
  Vector top = d_center.asVector() + d_normal*halfThick;
  Point lo(bot.x() - d_radius, bot.y() - d_radius,
	   bot.z() - d_radius);
  Point hi(top.x() + d_radius, top.y() + d_radius,
	   top.z() + d_radius);

  return Box(lo,hi);
}

//////////
// Find the particle count on the circular plane surface
// Create the particles on a circle on the x-y plane and then
// rotate them to the correct position and find if they are still
// in the patch. First particle is located at the center.
int 
PlaneShellPiece::returnParticleCount(const Patch* patch)
{
  // Get the bounding patch box
  Box b = patch->getBox();

  // The normal to the xy-plane
  Vector n0(0.0, 0.0, 1.0);
  
  // Angle of rotation
  double phi = acos(Dot(n0, d_normal)/(n0.length()*d_normal.length()));

  // Rotation axis
  Vector a = Cross(n0, d_normal);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Create particles
  int count = 0;
  double radInc = d_radius/(double) d_numRadius;
  for (int ii = 1; ii < d_numRadius+1; ++ii) {
    double currRadius = ii*radInc;
    int numCircum = (int) (2.0*M_PI*currRadius/radInc);
    double phiInc = 2.0*M_PI/(double) numCircum;
    for (int jj = 0; jj < numCircum; ++jj) {
      double phi = jj*phiInc; 

      // Create points on xy plane
      double x = currRadius*cos(phi);
      double y = currRadius*sin(phi);
      double z = 0.0;
     
      // Rotate to correct orientation
      // and translate to correct position
      Vector pp(x, y, z);
      pp = R*pp + d_center.asVector();
      Point p(pp);

      // If the patch contains the point, increment count
      if(b.contains(p)) ++count;
    }
  }

  return count;
}

//////////
// Create particles : uses the same algorithm as count particles.
// Create the particles on a circle on the x-y plane and then
// rotate them to the correct position and find if they are still
// in the patch. First particle is located at the center.
int 
PlaneShellPiece::createParticles(const Patch* patch,
				 ParticleVariable<Point>&  pos,
				 ParticleVariable<double>& vol,
				 ParticleVariable<double>& pThickTop,
				 ParticleVariable<double>& pThickBot,
				 ParticleVariable<Vector>& pNormal,
				 ParticleVariable<Vector>& psiz,
				 particleIndex start)
{
  cout << "Calling plane shell particle creator" << endl;

  // Get the bounding patch box
  Box b = patch->getBox();

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, d_normal)/(n0.length()*d_normal.length()));

  // Rotation axis
  Vector a = Cross(n0, d_normal);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Create particles
  int count = 0;
  double radInc = d_radius/(double) d_numRadius;
  for (int ii = 1; ii < d_numRadius+1; ++ii) {
    double prevRadius = (ii-1)*radInc;
    double currRadius = ii*radInc;
    int numCircum = (int) (2.0*M_PI*currRadius/radInc);
    double phiInc = 2.0*M_PI/(double) numCircum;
    double area = 0.5*phiInc*(currRadius*currRadius-prevRadius*prevRadius);
    for (int jj = 0; jj < numCircum; ++jj) {
      double phi = jj*phiInc; 
      double cosphi = cos(phi);
      double sinphi = sin(phi);

      // Create points on xy plane
      double x = currRadius*cosphi;
      double y = currRadius*sinphi;
      double z = 0.0;
     
      // Rotate points to correct orientation and
      // Translate to correct position
      Vector pp(x, y, z);
      pp = R*pp + d_center.asVector();
      Point p(pp);

      // If the patch contains the point, increment count
      if(b.contains(p)) {
        particleIndex pidx = start+count;
        pos[pidx] = p;
        vol[pidx] = d_thickness*area;
        psiz[pidx] = Vector(.5,.5,.5);
        pThickTop[pidx] = 0.5*d_thickness;
        pThickBot[pidx] = 0.5*d_thickness;
        pNormal[pidx]  = d_normal;
        count++;
      }
    }
  }
  
  return count;
}
