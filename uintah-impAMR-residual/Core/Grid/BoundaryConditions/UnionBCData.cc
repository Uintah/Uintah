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
  sfcx=mybc.sfcx;
  sfcy=mybc.sfcy;
  sfcz=mybc.sfcz;
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
  sfcx=rhs.sfcx;
  sfcy=rhs.sfcy;
  sfcz=rhs.sfcz;

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
    (*bc)->determineSFLimits(face,patch);
  }
    
  IntVector ln,hn;
  patch->getFaceNodes(face,0,ln,hn);
  for (NodeIterator bound(ln,hn);!bound.done();bound++) {
    Point p = patch->getLevel()->getNodePosition(*bound);
    if (inside(p)) 
      nb.push_back(*bound);
  }
  
  setNBoundaryIterator(nb);

  determineSFLimits(face,patch);

}

void UnionBCData::determineSFLimits(Patch::FaceType face, const Patch* patch)
{
#if 0
  cout << "UnionBC determineSFLimits()" << endl;
#endif
  vector<IntVector> sfx,sfy,sfz;
  for (vector<BCGeomBase*>::const_iterator bc = child.begin();
       bc != child.end(); ++bc) {
    vector<IntVector> x_itr, y_itr, z_itr;
    (*bc)->getSFCXIterator(x_itr);
    (*bc)->getSFCYIterator(y_itr);
    (*bc)->getSFCZIterator(z_itr);
    copy(x_itr.begin(),x_itr.end(),back_inserter(sfx));
    copy(y_itr.begin(),y_itr.end(),back_inserter(sfy));
    copy(z_itr.begin(),z_itr.end(),back_inserter(sfz));
  }
  setSFCXIterator(sfx);
  setSFCXIterator(sfy);
  setSFCXIterator(sfz);
#if 0
  for (vector<IntVector>::const_iterator it = sfcx.begin(); it != sfcx.end();
       ++it) 
    cout << "sfcx = " << *it << endl;
  for (vector<IntVector>::const_iterator it = sfcy.begin(); it != sfcy.end();
       ++it) 
    cout << "sfcy = " << *it << endl;
  for (vector<IntVector>::const_iterator it = sfcz.begin(); it != sfcz.end();
       ++it) 
    cout << "sfcz = " << *it << endl;
#endif
}
