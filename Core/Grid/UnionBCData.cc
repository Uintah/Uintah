#include <Packages/Uintah/Core/Grid/UnionBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

UnionBCData::UnionBCData() : BCGeomBase()
{
}


UnionBCData::UnionBCData(BCData& bc)
{
}

UnionBCData::~UnionBCData()
{
  for (int i = 0; i < (int) child.size(); i++) {
    delete child[i];
  }
  child.clear();
}

UnionBCData::UnionBCData(const UnionBCData& mybc)
{
  vector<BCGeomBase*>::const_iterator itr;
  for (itr=mybc.child.begin(); itr != mybc.child.end(); ++itr)
    child.push_back((*itr)->clone());

 
  boundary=mybc.boundary;
  interior=mybc.interior;
}

UnionBCData& UnionBCData::operator=(const UnionBCData& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  vector<BCGeomBase*>::const_iterator itr;
  for(itr=child.begin(); itr != child.end();++itr)
    delete *itr;

  child.clear();
  
  // copy the rhs to the lhs
  for (itr=rhs.child.begin(); itr != rhs.child.end();++itr)
    child.push_back((*itr)->clone());
  
  boundary = rhs.boundary;
  interior = rhs.interior;

  return *this;
}

UnionBCData* UnionBCData::clone()
{
  return scinew UnionBCData(*this);

}

void UnionBCData::addBCData(BCData& bc)
{
}

void UnionBCData::addBC(BoundCondBase* bc)
{
  
}

void UnionBCData::addBCData(BCGeomBase* bc)
{
  child.push_back(bc);
}

void UnionBCData::getBCData(BCData& bc) const
{
  child[0]->getBCData(bc);
}

bool UnionBCData::inside(const Point &p) const 
{
  for (int i = 0; i < (int)child.size(); i++) {
    if (child[i]->inside(p))
      return true;
  }
  return false;
}

