#include <Packages/Uintah/Core/Grid/CircleBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

CircleBCData::CircleBCData() 
{
  
}

CircleBCData::CircleBCData(ProblemSpecP &ps) 
{
  
}

CircleBCData::CircleBCData(BCData& bc)
  : d_bc(bc)
{
}

CircleBCData::CircleBCData(Point& p, double radius)
  : d_radius(radius), d_origin(p)
{
}

CircleBCData::~CircleBCData()
{
}

CircleBCData* CircleBCData::clone()
{
  return new CircleBCData(*this);
}

void CircleBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}

void CircleBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}


void CircleBCData::setBoundaryIterator(vector<IntVector>& b)
{
  boundary = b;
}

void CircleBCData::setInteriorIterator(vector<IntVector>& i)
{
  interior = i;
}

void CircleBCData::setSFCXIterator(vector<IntVector>& i)
{
  sfcx = i;
}

void CircleBCData::setSFCYIterator(vector<IntVector>& i)
{
  sfcy = i;
}

void CircleBCData::setSFCZIterator(vector<IntVector>& i)
{
  sfcz = i;
}

void CircleBCData::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void CircleBCData::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}

void CircleBCData::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void CircleBCData::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void CircleBCData::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}


bool CircleBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else
    return true;
}

