#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
using namespace SCIRun;
using namespace Uintah;

BCDataArray::BCDataArray() 
{
}

BCDataArray::BCDataArray(ProblemSpecP &ps) 
{
  //BoundCondFactory::create(ps,child);
  
}

BCDataArray::BCDataArray(BCData& bc)
{
  //BoundCondFactory::create(ps,child);
  
}

BCDataArray::~BCDataArray()
{
  for (int i = 0; i < (int) child.size(); i++) {
    delete child[i];
  }
  child.clear();
}

BCDataArray::BCDataArray(const BCDataArray& mybc)
{
  for (int i = 0; i < (int) mybc.child.size(); i++) {
    BCDataBase* bc;
    bc = mybc.child[i]->clone();
    child.push_back(bc);
  }
}

BCDataArray& BCDataArray::operator=(const BCDataArray& rhs)
{
  if (this == &rhs)  return *this;

  child.clear();

  for (int i = 0; i < (int) child.size(); i++) {
    delete child[i];
  }
  
  for (int i = 0; i < (int) rhs.child.size(); i++) {
    BCDataBase* bc;
    bc = rhs.child[i]->clone();
    child.push_back(bc);
  }
  return *this;
}

BCDataArray* BCDataArray::clone()
{
  return new BCDataArray(*this);

}

void BCDataArray::addBCData(BCData& bc)
{
}

void BCDataArray::addBCData(BCDataBase* bc)
{
  child.push_back(bc);
}

void BCDataArray::getBCData(BCData& bc, int i) const
{
  child[i]->getBCData(bc);
}

void BCDataArray::setBoundaryIterator(vector<IntVector>& b,int i)
{
  child[i]->setBoundaryIterator(b);
}

void BCDataArray::setInteriorIterator(vector<IntVector>& i,int ii)
{
  child[ii]->setInteriorIterator(i);
}

void BCDataArray::setSFCXIterator(vector<IntVector>& i,int ii)
{
  child[ii]->setSFCXIterator(i);
}

void BCDataArray::setSFCYIterator(vector<IntVector>& i,int ii)
{
  child[ii]->setSFCYIterator(i);
}

void BCDataArray::setSFCZIterator(vector<IntVector>& i,int ii)
{
  child[ii]->setSFCZIterator(i);
}

void BCDataArray::getBoundaryIterator(vector<IntVector>& b,int i) const
{
  child[i]->getBoundaryIterator(b);
}

void BCDataArray::getInteriorIterator(vector<IntVector>& i,int ii) const
{
  child[ii]->getInteriorIterator(i);
}

void BCDataArray::getSFCXIterator(vector<IntVector>& i,int ii) const
{
  child[ii]->getSFCXIterator(i);
}

void BCDataArray::getSFCYIterator(vector<IntVector>& i,int ii) const
{
  child[ii]->getSFCYIterator(i);
}

void BCDataArray::getSFCZIterator(vector<IntVector>& i,int ii) const
{
  child[ii]->getSFCZIterator(i);
}

int BCDataArray::getNumberChildren() const
{
  return child.size();
}

BCDataBase* BCDataArray::getChild(int i) const
{
  return child[i];
}

