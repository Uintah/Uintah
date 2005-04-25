#include <Packages/Uintah/Core/GeometryPiece/SmoothGeomPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


SmoothGeomPiece::SmoothGeomPiece()
{
  d_dx = 1.0;
}

SmoothGeomPiece::~SmoothGeomPiece()
{
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle locations */
//////////////////////////////////////////////////////////////////////
vector<Point>* 
SmoothGeomPiece::getPoints()
{
  return &d_points;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle volumes */
//////////////////////////////////////////////////////////////////////
vector<double>* 
SmoothGeomPiece::getVolume()
{
  return &d_volume;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle forces */
//////////////////////////////////////////////////////////////////////
vector<Vector>* 
SmoothGeomPiece::getForces()
{
  return &d_forces;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle fiber directions */
//////////////////////////////////////////////////////////////////////
vector<Vector>* 
SmoothGeomPiece::getFiberDirs()
{
  return &d_fiberdirs;
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle locations */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deletePoints()
{
  d_points.clear();
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle volumes */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deleteVolume()
{
  d_volume.clear();
}

void 
SmoothGeomPiece::writePoints(const string& f_name, const string& var)
{
  cout << "Not implemented : " << f_name << "." << var 
            << " output " << endl;
}

int 
SmoothGeomPiece::returnPointCount() const
{
  return d_points.size();
}

void 
SmoothGeomPiece::setParticleSpacing(double dx)
{
  d_dx = dx;
}
