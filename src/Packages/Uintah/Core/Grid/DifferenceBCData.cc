#include <Packages/Uintah/Core/Grid/DifferenceBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;
using std::vector;

DifferenceBCData::DifferenceBCData(BCGeomBase* p1,BCGeomBase* p2)
  : left(p1->clone()), right(p2->clone())
{
}

DifferenceBCData::DifferenceBCData(const DifferenceBCData& rhs)
{
  left=rhs.left->clone();
  right=rhs.right->clone();

  boundary=rhs.boundary;
  interior=rhs.interior;
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;

}

DifferenceBCData& DifferenceBCData::operator=(const DifferenceBCData& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  delete right;
  delete left;

  // Copy the rhs to the lhs

  left = rhs.left->clone();
  right = rhs.right->clone();

  boundary = rhs.boundary;
  interior = rhs.interior;
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
 
  return *this;
}

DifferenceBCData::~DifferenceBCData()
{
  delete left;
  delete right;
}

DifferenceBCData* DifferenceBCData::clone()
{
  return scinew DifferenceBCData(*this);
}

void DifferenceBCData::addBCData(BCData& bc)
{

}


void DifferenceBCData::addBC(BoundCondBase* bc)
{

}

void DifferenceBCData::getBCData(BCData& bc) const
{
  left->getBCData(bc);
}

bool DifferenceBCData::inside(const Point &p) const 
{
  return (left->inside(p) && !right->inside(p));
}

