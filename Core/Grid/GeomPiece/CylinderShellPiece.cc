#include <Packages/Uintah/Core/Grid/GeomPiece/CylinderShellPiece.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;
using namespace SCIRun;

///////////
// Constructor
CylinderShellPiece::CylinderShellPiece(ProblemSpecP& ps)
{
  ps->require("top",d_top);
  ps->require("bottom",d_bottom);
  ps->require("radius",d_radius);
  ps->require("thickness",d_thickness);
  ps->require("num_axis",d_numAxis);
  ps->require("num_circum",d_numCircum);
  
  double near_zero = 1e-100;
  Vector axis = d_top - d_bottom;
  
  if (axis.length() < near_zero) 
    SCI_THROW(ProblemSetupException("Error:CylinderShell:Axis length = 0.0"));
  if (d_radius <= 0.0) 
    SCI_THROW(ProblemSetupException("Error:CylinderShell:Radius <= 0.0"));
  if (d_thickness <= 0.0)
    SCI_THROW(ProblemSetupException("Error:CylinderShell:Thickness <= 0.0"));
  if (d_numAxis < 1 || d_numCircum < 1)
    SCI_THROW(ProblemSetupException("Error:CylinderShell:Divisions < 1"));
}

///////////
// Destructor
CylinderShellPiece::~CylinderShellPiece()
{
}

///////////
// Point inside cylinder
bool 
CylinderShellPiece::inside(const Point& p) const
{
  Vector axis = d_top-d_bottom;  
  Vector tobot = p-d_bottom;
  double h = Dot(tobot, axis);
  double height2 = axis.length2();
  // Above or below the cylinder
  if(h < 0.0 || h > height2) return false; 
  double area = Cross(axis, tobot).length2();
  double d = area/height2;
  if( d > d_radius*d_radius) return false;
  return true;
}

///////////
// Bounding box
Box 
CylinderShellPiece::getBoundingBox() const
{
  Point lo(d_bottom.x() - d_radius, d_bottom.y() - d_radius,
	   d_bottom.z() - d_radius);
  Point hi(d_top.x() + d_radius, d_top.y() + d_radius,
	   d_top.z() + d_radius);
  return Box(lo,hi);
}

/////////////////////////////////////////////////////////////////////////////
/*! Count particles on cylinder
   The particles are created at the bottom plane and then
   copies are rotated and translated to get the final count
   Particles are counted only if they are inside the patch */
/////////////////////////////////////////////////////////////////////////////
int 
CylinderShellPiece::returnParticleCount(const Patch* patch)
{
  // Get the patch box
  Box b = patch->getBox();

  // Find the direction of the normal to the base 
  Vector normal = d_top - d_bottom;
  double length = normal.length();
  normal /= length;

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, normal)/(n0.length()*normal.length()));

  // Rotation axis
  Vector a = Cross(n0, normal);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Create particles 
  int count = 0;
  double incAxis = length/(double) d_numAxis;
  double incCircum = 2.0*M_PI/(double) d_numCircum;
  for (int ii = 0; ii < d_numAxis+1; ++ii) {
    double currHeight = ii*incAxis;
    Vector currCenter = d_bottom.asVector() + normal*currHeight;
    for (int jj = 0; jj < d_numCircum; ++jj) {
      double currPhi = jj*incCircum;

      // Create points on xy plane
      double x = d_radius*cos(currPhi);
      double y = d_radius*sin(currPhi);
      double z = 0.0;
     
      // Rotate points to correct orientation and
      // Translate to correct position
      Vector pp(x, y, z);
      pp = R*pp + currCenter;
      Point p(pp);

      // If the patch contains the point, increment count
      if(b.contains(p)) ++count;
    }
  } 
  return count;
}

///////////
/////////////////////////////////////////////////////////////////////////////
/*! Create particles on cylinder
   Same algorithm as particle count
   The particles are created at the bottom plane and then
   copies are rotated and translated to get the final count
   Particles are counted only if they are inside the patch */
/////////////////////////////////////////////////////////////////////////////
int 
CylinderShellPiece::createParticles(const Patch* patch,
				  ParticleVariable<Point>&  pos,
				  ParticleVariable<double>& vol,
				  ParticleVariable<double>& pThickTop,
				  ParticleVariable<double>& pThickBot,
				  ParticleVariable<Vector>& pNormal,
				  ParticleVariable<Vector>& psiz,
				  particleIndex start)
{
  // Get the patch box
  Box b = patch->getBox();

  // Find the direction of the normal to the base 
  Vector normal = d_top - d_bottom;
  double length = normal.length();
  normal /= length;

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, normal)/(n0.length()*normal.length()));

  // Rotation axis
  Vector a = Cross(n0, normal);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Create particles 
  int count = 0;
  double incAxis = length/(double) d_numAxis;
  double incCircum = 2.0*M_PI/(double) d_numCircum;
  for (int ii = 0; ii < d_numAxis+1; ++ii) {
    double currHeight = ii*incAxis;
    Vector currCenter = d_bottom.asVector() + normal*currHeight;
    double axisThickness = incAxis;
    if (ii == 0 || ii == d_numAxis) axisThickness = 0.5*incAxis;
    for (int jj = 0; jj < d_numCircum; ++jj) {
      double currPhi = jj*incCircum;
      double cosphi = cos(currPhi);
      double sinphi = sin(currPhi);

      // Create points on xy plane
      double x = d_radius*cosphi;
      double y = d_radius*sinphi;
      double z = 0.0;
     
      // Rotate points to correct orientation and
      // Translate to correct position
      Vector pp(x, y, z);
      pp = R*pp + currCenter;
      Point p(pp);

      // Create the particle if it is in the patch
      if(b.contains(p)){
        particleIndex pidx = start+count;
        pos[pidx] = p;
        vol[pidx] = incCircum*d_radius*d_thickness*axisThickness;
        psiz[pidx] = Vector(.5,.5,.5);
        pThickTop[pidx] = 0.5*d_thickness;
        pThickBot[pidx] = 0.5*d_thickness;

        // Create the normal to the circle at the points
        // and rotate to correct orientation
        pNormal[pidx] = Vector(cosphi, sinphi, 0);
        pNormal[pidx] = R*pNormal[pidx];

        count++;
      }
    } 
  }
  return count;
}
