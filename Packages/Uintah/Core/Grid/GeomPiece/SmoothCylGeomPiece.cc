#include <Packages/Uintah/Core/Grid/GeomPiece/SmoothCylGeomPiece.h>
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
SmoothCylGeomPiece::SmoothCylGeomPiece(ProblemSpecP& ps)
{
  setName("smoothcyl");
  ps->require("bottom", d_bottom);
  ps->require("top", d_top);
  if ((d_top-d_bottom).length2() <= 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Check data in input"));

  ps->require("radius", d_radius);
  if (d_radius <= 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Radius < 0"));

  ps->require("num_radial", d_numRadial);
  if (d_numRadial < 1)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Radial Divs < 1"));

  ps->require("num_axial", d_numAxial);
  if (d_numAxial < 1)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Axial Divs < 1"));

  d_thickness = d_radius;
  ps->get("thickness", d_thickness);
  if (d_thickness > d_radius)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Thickness > Radius"));

  d_capThick = 0.0;
  ps->get("endcap_thickness", d_capThick);
  if (d_capThick < 0.0)
    SCI_THROW(ProblemSetupException("SmoothCylGeom: Cap Thickness < 0.0"));
}

//////////
// Destructor
SmoothCylGeomPiece::~SmoothCylGeomPiece()
{
}

/////////////////////////////////////////////////////////////////////////////
/*! Find if a point is inside the cylinder or end caps */
/////////////////////////////////////////////////////////////////////////////
bool 
SmoothCylGeomPiece::inside(const Point& p) const
{
  bool isInside = true;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double length = axis.length();
  Vector capAxis = axis*(d_capThick/length);

  // a) Check is the point is within the solid composite cylinder
  Vector bot = d_bottom.asVector() - capAxis;
  Vector top = d_top.asVector() + capAxis;
  Vector fullAxis = top - bot;
  double height  = length + 2*d_capThick;
  double height2 = height*height;
  Vector pToBot = p.asVector() - bot;
  double h = Dot(pToBot, fullAxis);
  if(h < 0.0 || h > height2) isInside = false;
  double area = Cross(fullAxis, pToBot).length2();
  double d = area/height2;
  if( d > d_radius*d_radius) isInside = false;

  // b) Find if the point is outside the inner cylinder
  if (isInside) {
    pToBot = p - d_bottom;
    area = Cross(axis, pToBot).length2();
    d = area/(length*length);
    double innerRad = d_radius - d_thickness;
    if(!(d > innerRad*innerRad)) isInside = false;
  }
  return isInside;
}

//////////
// Find the bounding box for the cylinder
Box 
SmoothCylGeomPiece::getBoundingBox() const
{
  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  Vector capAxis = axis*(d_capThick/axis.length());

  Vector bot = d_bottom.asVector() - capAxis;
  Vector top = d_top.asVector() + capAxis;
  Point lo(bot.x() - d_radius, bot.y() - d_radius,
	   bot.z() - d_radius);
  Point hi(top.x() + d_radius, top.y() + d_radius,
	   top.z() + d_radius);

  return Box(lo,hi);
}

//////////////////////////////////////////////////////////////////////////
/*! Count particles */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::returnParticleCount(const Patch* patch)
{
  bool doCreate = false;
  ParticleVariable<Point> pos;
  ParticleVariable<double> vol;
  ParticleVariable<Vector> psiz;
  particleIndex start = 0;
  int totCount = 0;
  if (d_capThick > 0.0) {
    int count = createOrCountEndCapParticles(patch, pos, vol, psiz, start, 
                                             doCreate);
    totCount += count;
  }
  if (d_thickness < d_radius) {
    int count = createOrCountHollowCylParticles(patch, pos, vol, psiz, start, 
						doCreate);
    totCount += count;
  } else {
    int count = createOrCountSolidCylParticles(patch, pos, vol, psiz, start, 
					       doCreate);
    totCount += count;
  }
  cout << "Particle Count = " << totCount << endl;
  return totCount;
}

//////////////////////////////////////////////////////////////////////////
/*! Create particles */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createParticles(const Patch* patch,
				    ParticleVariable<Point>&  pos,
				    ParticleVariable<double>& vol,
				    ParticleVariable<Vector>& psiz,
				    particleIndex start)
{
  bool doCreate = true;
  int totCount = 0;
  if (d_capThick > 0.0) {
    int count = createOrCountEndCapParticles(patch, pos, vol, psiz, start, 
                                             doCreate);
    start += count;
    totCount += count;
  }
  if (d_thickness < d_radius) {
    int count = createOrCountHollowCylParticles(patch, pos, vol, psiz, start, 
						doCreate);
    start += count;
    totCount += count;
  } else {
    int count = createOrCountSolidCylParticles(patch, pos, vol, psiz, start, 
					       doCreate);
    start += count;
    totCount += count;
  }
  return totCount;
}

//////////////////////////////////////////////////////////////////////////
/*! Create the particles on a circle on the x-y plane and then
  rotate them to the correct position and find if they are still
  in the patch. First particle is located at the center. */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createOrCountEndCapParticles(const Patch* patch,
						 ParticleVariable<Point>&  pos,
						 ParticleVariable<double>& vol,
						 ParticleVariable<Vector>& psiz,
						 particleIndex start,
						 bool doCreate)
{
  cout << "Creating particles for the End Caps" << endl;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double axislen = axis.length();
  axis /= axislen;

  // Get the bounding patch box
  Box b = patch->getBox();

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, axis));

  // Rotation axis
  Vector a = Cross(n0, axis);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Initialize count of the number of particles
  int count = 0;

  // Calculate the radial and axial material point spacing
  double axisInc = axislen/(double) d_numAxial;
  int numCapAxial = int(d_capThick/axisInc);
  double radInc = d_radius/(double) d_numRadial;

  // Create particles for the bottom end cap
  double currZ = d_bottom.z() - 0.5*axisInc;
  for (int kk = 0; kk < numCapAxial; ++kk) {
    Vector currCenter = d_bottom.asVector() - axis*currZ;
    for (int ii = 1; ii < d_numRadial+1; ++ii) {
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
	double z = 0;
     
	// Rotate points to correct orientation and
	// Translate to correct position
	Vector pp(x, y, z);
	pp = R*pp + currCenter;
	Point p(pp);

	// If the patch contains the point, increment count
	if(b.contains(p)) {
          if (doCreate) {
	    particleIndex pidx = start+count;
	    pos[pidx] = p;
	    vol[pidx] = axisInc*area;
	    psiz[pidx] = Vector(.5,.5,.5);
          } 
	  count++;
	}
      }
    }
    currZ -= axisInc;
  }
  
  // Create particles for the top end cap
  currZ = d_top.z() + 0.5*axisInc;
  for (int kk = 0; kk < numCapAxial; ++kk) {
    Vector currCenter = d_top.asVector() + axis*currZ;
    for (int ii = 1; ii < d_numRadial+1; ++ii) {
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
	double z = 0;
     
	// Rotate points to correct orientation and
	// Translate to correct position
	Vector pp(x, y, z);
	pp = R*pp + currCenter;
	Point p(pp);

	// If the patch contains the point, increment count
	if(b.contains(p)) {
          if (doCreate) {
	    particleIndex pidx = start+count;
	    pos[pidx] = p;
	    vol[pidx] = axisInc*area;
	    psiz[pidx] = Vector(.5,.5,.5);
          } 
	  count++;
	}
      }
    }
    currZ += axisInc;
  }
  
  return count;
}

//////////////////////////////////////////////////////////////////////////
/*! Create the particles on a circle on the x-y plane and then
  rotate them to the correct position and find if they are still
  in the patch. First particle is located at the center. */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createOrCountSolidCylParticles(const Patch* patch,
						 ParticleVariable<Point>& pos,
						 ParticleVariable<double>& vol,
						 ParticleVariable<Vector>& psiz,
						 particleIndex start,
						 bool doCreate)
{
  cout << "Creating particles for the Solid Cylinder" << endl;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double length = axis.length();
  axis /= length;

  // Get the bounding patch box
  Box b = patch->getBox();

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, axis));

  // Rotation axis
  Vector a = Cross(n0, axis);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Initialize count of the number of particles
  int count = 0;

  // Calculate the radial and axial material point spacing
  double axisInc = length/(double) d_numAxial;
  double radInc = d_radius/(double) d_numRadial;

  // Create particles for the bottom end cap
  double currZ = d_bottom.z() + 0.5*axisInc;
  for (int kk = 0; kk < d_numAxial; ++kk) {
    Vector currCenter = d_bottom.asVector() + axis*currZ;
    for (int ii = 1; ii < d_numRadial+1; ++ii) {
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
	double z = 0;
	Vector pp(x, y, z);
     
	// Rotate points to correct orientation and
	// Translate to correct position
	pp = R*pp + currCenter;

	// If the patch contains the point, increment count
	Point p(pp);
	if(b.contains(p)) {
          if (doCreate) {
	    particleIndex pidx = start+count;
	    pos[pidx] = p;
	    vol[pidx] = axisInc*area;
	    psiz[pidx] = Vector(.5,.5,.5);
          } 
	  count++;
	}
      }
    }
    currZ += axisInc;
  }
  
  return count;
}

//////////////////////////////////////////////////////////////////////////
/*! Create the particles on a circle on the x-y plane and then
  rotate them to the correct position and find if they are still
  in the patch. */
//////////////////////////////////////////////////////////////////////////
int 
SmoothCylGeomPiece::createOrCountHollowCylParticles(const Patch* patch,
						 ParticleVariable<Point>&  pos,
						 ParticleVariable<double>& vol,
						 ParticleVariable<Vector>& psiz,
						 particleIndex start,
						 bool doCreate)
{
  cout << "Creating particles for the Hollow Cylinder" << endl;

  // Find the vector along the axis of the cylinder
  Vector axis = d_top - d_bottom;
  double length = axis.length();
  axis = axis/length;

  // Get the bounding patch box
  Box b = patch->getBox();

  // Angle of rotation
  Vector n0(0.0, 0.0, 1.0); // The normal to the xy-plane
  double phi = acos(Dot(n0, axis));

  // Rotation axis
  Vector a = Cross(n0, axis);
  a /= (a.length()+1.0e-100);

  // Create Rotation matrix 
  Matrix3 R(phi, a);

  // Initialize count of the number of particles
  int count = 0;

  // Calculate the radial and axial material point spacing
  double axisInc = length/(double) d_numAxial;
  double radInc = d_radius/(double) d_numRadial;
  int numThick = (int)(d_thickness/radInc);
  double innerRad = d_radius - d_thickness;

  // Create particles for the hollow cylinder
  double currZ = d_bottom.z() + 0.5*axisInc;
  for (int kk = 0; kk < d_numAxial; ++kk) {
    Vector currCenter = d_bottom.asVector() + axis*currZ;
    for (int ii = 1; ii < numThick+1; ++ii) {
      double prevRadius = innerRad + (ii-1)*radInc;
      double currRadius = innerRad + ii*radInc;
      double radLoc = 0.5*(prevRadius+currRadius);
      int numCircum = (int) (2.0*M_PI*currRadius/radInc);
      double phiInc = 2.0*M_PI/(double) numCircum;
      double area = 0.5*phiInc*(currRadius*currRadius-prevRadius*prevRadius);
      for (int jj = 0; jj < numCircum; ++jj) {
	double phi = jj*phiInc; 
	double cosphi = cos(phi);
	double sinphi = sin(phi);

	// Create points on xy plane
	double x = radLoc*cosphi;
	double y = radLoc*sinphi;
	double z = 0;
     
	// Rotate points to correct orientation and
	// Translate to correct position
	Vector pp(x, y, z);
	pp = R*pp + currCenter;
	Point p(pp);

	// If the patch contains the point, increment count
	if(b.contains(p)) {
          if (doCreate) {
	    particleIndex pidx = start+count;
	    pos[pidx] = p;
	    vol[pidx] = axisInc*area;
	    psiz[pidx] = Vector(.5,.5,.5);
          } 
	  count++;
	}
      }
    }
    currZ += axisInc;
  }
  
  return count;
}
