#include <Packages/Uintah/Core/Grid/SideBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;
SideBCData::SideBCData() 
{
  
}


SideBCData::SideBCData(BCData& bc) 
  : d_bc(bc)
{
}

SideBCData::~SideBCData()
{
}

#if 0
SideBCData::SideBCData(const SideBCData& mybc)
{
  d_bc=mybc.d_bc;
  boundary=mybc.boundary;
  interior=mybc.interior;
  sfcx=mybc.sfcx;
  sfcy=mybc.sfcy;
  sfcz=mybc.sfcz;

}


SideBCData& SideBCData::operator=(const SideBCData& rhs)
{
  if (this == &rhs)
    return *this;

  d_bc = rhs.d_bc;
  boundary = rhs.boundary;
  interior = rhs.interior;
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;

  return *this;
}
#endif

SideBCData* SideBCData::clone()
{
  return scinew SideBCData(*this);

}
void SideBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}

void SideBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}


void SideBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool SideBCData::inside(const Point &p) const 
{
  return true;
}

