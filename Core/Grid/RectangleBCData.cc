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


void RectangleBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}

void RectangleBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool RectangleBCData::inside(const Point &p) const 
{
  if (p == Max(p,d_min) && p == Min(p,d_max) )
    return true;
  else 
    return false;
}

