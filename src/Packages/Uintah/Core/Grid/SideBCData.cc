#include <Packages/Uintah/Core/Grid/SideBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;
SideBCData::SideBCData() 
{
  
}

SideBCData::SideBCData(ProblemSpecP &ps) 
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
  return new SideBCData(*this);

}
void SideBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}

void SideBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

void SideBCData::setBoundaryIterator(vector<IntVector>& b)
{
  boundary=b;
}

void SideBCData::setInteriorIterator(vector<IntVector>& i)
{
  interior=i;
}

void SideBCData::setSFCXIterator(vector<IntVector>& i)
{
  sfcx=i;
}

void SideBCData::setSFCYIterator(vector<IntVector>& i)
{
  sfcy=i;
}

void SideBCData::setSFCZIterator(vector<IntVector>& i)
{
  sfcz=i;
}


void SideBCData::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void SideBCData::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}

void SideBCData::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void SideBCData::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void SideBCData::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}



bool SideBCData::inside(const Point &p) const 
{
  return true;
}

