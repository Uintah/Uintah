#include <Packages/Uintah/Core/Grid/CircleBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

CircleBCData::CircleBCData() 
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
  return scinew CircleBCData(*this);
}

void CircleBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void CircleBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void CircleBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool CircleBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else
    return true;
}

