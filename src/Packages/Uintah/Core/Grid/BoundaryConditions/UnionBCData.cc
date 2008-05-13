#include <Packages/Uintah/Core/Grid/BoundaryConditions/UnionBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

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
  for (vector<BCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc)
    delete (*bc);
  
  child.clear();
}

UnionBCData::UnionBCData(const UnionBCData& mybc)
{
  vector<BCGeomBase*>::const_iterator itr;
  for (itr=mybc.child.begin(); itr != mybc.child.end(); ++itr)
    child.push_back((*itr)->clone());

 
  boundary=mybc.boundary;
  nboundary=mybc.nboundary;
#if 0
  sfcx=mybc.sfcx;
  sfcy=mybc.sfcy;
  sfcz=mybc.sfcz;
#endif
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
  nboundary = rhs.nboundary;
#if 0
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;
#endif

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
  for (vector<BCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i){
    if ((*i)->inside(p))
      return true;
  }
  return false;
}

void UnionBCData::print()
{
  cout << "Geometry type = " << typeid(this).name() << endl;
  for (vector<BCGeomBase*>::const_iterator i = child.begin(); i != child.end();
       ++i)
    (*i)->print();

}

void UnionBCData::determineIteratorLimits(Patch::FaceType face, 
					  const Patch* patch, 
					  vector<Point>& test_pts)
{
#if 0
  cout << "UnionBC determineIteratorLimits()" << endl;
#endif
  IntVector l,h;
  patch->getFaceCells(face,0,l,h);

  vector<IntVector> b,nb;
  vector<Point>::iterator pts;
  pts = test_pts.begin();
  for (CellIterator bound(l,h); !bound.done(); bound++,pts++) 
    if (inside(*pts))
      b.push_back(*bound);

  setBoundaryIterator(b);
#if 0
  cout << "Size of boundary = " << boundary.size() << endl;
#endif
  // Need to determine the boundary iterators for each separate bc.
  for (vector<BCGeomBase*>::const_iterator bc = child.begin();  
       bc != child.end(); ++bc) {
    pts = test_pts.begin();
    vector<IntVector> boundary_itr;
    for (CellIterator bound(l,h); !bound.done(); bound++, pts++) 
      if ( (*bc)->inside(*pts))
	boundary_itr.push_back(*bound);
#if 0
    cout << "Size of boundary_itr = " << boundary_itr.size() << endl;
#endif
    (*bc)->setBoundaryIterator(boundary_itr);
  }
    
  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  for (NodeIterator bound(ln,hn);!bound.done();bound++) {
    Point p = patch->getLevel()->getNodePosition(*bound);
    if (inside(p)) 
      nb.push_back(*bound);
  }
  
  setNBoundaryIterator(nb);

}

