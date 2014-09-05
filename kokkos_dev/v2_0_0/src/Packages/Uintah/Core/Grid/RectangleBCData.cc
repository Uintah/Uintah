#include <Packages/Uintah/Core/Grid/RectangleBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

RectangleBCData::RectangleBCData() 
{
  
}

RectangleBCData::RectangleBCData(ProblemSpecP &ps) 
{
  
}

RectangleBCData::RectangleBCData(BCData& bc)
  : d_bc(bc)
{
}

RectangleBCData::RectangleBCData(Point& low, Point& up)
  : d_min(low), d_max(up)
{
}

RectangleBCData::~RectangleBCData()
{
}

RectangleBCData* RectangleBCData::clone()
{
  return new RectangleBCData(*this);
}

void RectangleBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}

void RectangleBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

void RectangleBCData::setBoundaryIterator(vector<IntVector>& b)
{
  boundary = b;
}

void RectangleBCData::setInteriorIterator(vector<IntVector>& i)
{
  interior = i;
}

void RectangleBCData::setSFCXIterator(vector<IntVector>& i)
{
  sfcx = i;
}

void RectangleBCData::setSFCYIterator(vector<IntVector>& i)
{
  sfcy = i;
}

void RectangleBCData::setSFCZIterator(vector<IntVector>& i)
{
  sfcz = i;
}

void RectangleBCData::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void RectangleBCData::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}

void RectangleBCData::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void RectangleBCData::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void RectangleBCData::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}


bool RectangleBCData::inside(const Point &p) const 
{
  if (p == Max(p,d_min) && p == Min(p,d_max) )
    return true;
  else 
    return false;
}

