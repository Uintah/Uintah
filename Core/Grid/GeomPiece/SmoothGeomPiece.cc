#include <Packages/Uintah/Core/Grid/GeomPiece/SmoothGeomPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


SmoothGeomPiece::SmoothGeomPiece()
{
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
