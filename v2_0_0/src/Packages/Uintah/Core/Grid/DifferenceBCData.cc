#include <Packages/Uintah/Core/Grid/DifferenceBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;

DifferenceBCData::DifferenceBCData(ProblemSpecP &ps) 
{
  //BoundCondFactory::create(ps,child);
  
}

DifferenceBCData::DifferenceBCData(BCDataBase* p1,BCDataBase* p2)
  : left(p1), right(p2)
{
}

DifferenceBCData::DifferenceBCData(const DifferenceBCData& rhs)
{
  BCDataBase* l = rhs.left->clone();
  BCDataBase* r = rhs.right->clone();
  left = l;
  right = r;

  boundary=rhs.boundary;
  interior=rhs.interior;
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;

}

DifferenceBCData& DifferenceBCData::operator=(const DifferenceBCData& rhs)
{
  BCDataBase* l = rhs.left->clone();
  BCDataBase* r = rhs.right->clone();

  left = l;
  right = r;

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
  return new DifferenceBCData(*this);
}

void DifferenceBCData::addBCData(BCData& bc)
{

}

void DifferenceBCData::getBCData(BCData& bc) const
{
  left->getBCData(bc);
}

void DifferenceBCData::setBoundaryIterator(vector<IntVector>& b) 
{
  boundary = b;
}

void DifferenceBCData::setInteriorIterator(vector<IntVector>& i) 
{
  interior = i;
}

void DifferenceBCData::setSFCXIterator(vector<IntVector>& i)
{
  sfcx = i;
}

void DifferenceBCData::setSFCYIterator(vector<IntVector>& i)
{
  sfcy = i;
}

void DifferenceBCData::setSFCZIterator(vector<IntVector>& i)
{
  sfcz = i;
}

void DifferenceBCData::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void DifferenceBCData::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}

void DifferenceBCData::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void DifferenceBCData::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void DifferenceBCData::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}



bool DifferenceBCData::inside(const Point &p) const 
{
  return (left->inside(p) && !right->inside(p));
}

