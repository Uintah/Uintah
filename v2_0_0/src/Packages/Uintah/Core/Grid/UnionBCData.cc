#include <Packages/Uintah/Core/Grid/UnionBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

UnionBCData::UnionBCData() 
{
}

UnionBCData::UnionBCData(ProblemSpecP &ps) 
{
  //BoundCondFactory::create(ps,child);
  
}

UnionBCData::UnionBCData(BCData& bc)
{
  //BoundCondFactory::create(ps,child);
  
}

UnionBCData::~UnionBCData()
{
  for (int i = 0; i < (int) child.size(); i++) {
    delete child[i];
  }
}

UnionBCData::UnionBCData(const UnionBCData& mybc)
{
  for (int i = 0; i < (int) mybc.child.size(); i++) {
    BCDataBase* bc;
    bc = mybc.child[i]->clone();
    child.push_back(bc);
  }

  boundary=mybc.boundary;
  interior=mybc.interior;
  sfcx=mybc.sfcx;
  sfcy=mybc.sfcy;
  sfcz=mybc.sfcz;

}

UnionBCData& UnionBCData::operator=(const UnionBCData& rhs)
{
  for (int i = 0; i < (int) rhs.child.size(); i++) {
    BCDataBase* bc;
    bc = rhs.child[i]->clone();
    child.push_back(bc);
  }
  boundary = rhs.boundary;
  interior = rhs.interior;
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;

  return *this;
}

UnionBCData* UnionBCData::clone()
{
  return new UnionBCData(*this);

}

void UnionBCData::addBCData(BCData& bc)
{
}

void UnionBCData::addBCData(BCDataBase* bc)
{
  child.push_back(bc);
}

void UnionBCData::getBCData(BCData& bc) const
{
  child[0]->getBCData(bc);
}

void UnionBCData::setBoundaryIterator(vector<IntVector>& b)
{
  boundary = b;
}

void UnionBCData::setInteriorIterator(vector<IntVector>& i)
{
  interior = i;
}

void UnionBCData::setSFCXIterator(vector<IntVector>& i)
{
  sfcx = i;
}

void UnionBCData::setSFCYIterator(vector<IntVector>& i)
{
  sfcy = i;
}

void UnionBCData::setSFCZIterator(vector<IntVector>& i)
{
  sfcz = i;
}

void UnionBCData::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void UnionBCData::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}

void UnionBCData::getSFCXIterator(vector<IntVector>& i) const
{
  i = sfcx;
}

void UnionBCData::getSFCYIterator(vector<IntVector>& i) const
{
  i = sfcy;
}

void UnionBCData::getSFCZIterator(vector<IntVector>& i) const
{
  i = sfcz;
}


bool UnionBCData::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child.size(); i++) {
    if (child[i]->inside(p))
      return true;
  }
  return false;
}

